<?php

namespace App\Http\Controllers;

use App\Models\FaceProfile;
use App\Models\FaceRecognitionLog;
use App\Services\FaceRecognitionService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Schema;

class FaceRecognitionController extends Controller
{
    public function __construct(
        protected FaceRecognitionService $recognitionService
    ) {}

    /**
     * List all registered face profiles.
     */
    public function getProfiles(): JsonResponse
    {
        $profiles = FaceProfile::orderBy('created_at', 'desc')->get()->map(function ($profile) {
            $embedding = $profile->embedding ?? [];
            $templates = is_array($profile->templates) ? $profile->templates : [];
            return [
                'id' => $profile->id,
                'identifier' => $profile->identifier,
                'name' => $profile->name,
                'role' => $profile->role,
                'gender' => $profile->gender,
                'dimensions' => count($embedding),
                'storage_bytes' => count($embedding) * 4,
                'embedding' => $embedding,
                'templates' => $templates,
                'embedding_preview' => array_slice($embedding, 0, 5),
                'sample_count' => max((int)$profile->sample_count, count($templates) ?: 1),
                'template_count' => count($templates) ?: 1,
                'last_seen_at' => $profile->last_seen_at?->diffForHumans(),
                'created_at' => $profile->created_at->format('Y-m-d H:i:s'),
            ];
        });

        return response()->json([
            'status' => 'success',
            'count' => $profiles->count(),
            'data' => $profiles,
        ]);
    }

    /**
     * Register a new face profile WITHOUT saving photo.
     * Only stores name, identifier, role, and the 128D mathematical vector.
     */
    public function register(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'name' => 'required|string|max:100',
            'identifier' => 'nullable|string|max:50',
            'role' => 'nullable|string|max:50',
            'gender' => 'nullable|string|max:20',
            'embedding' => 'nullable|array',
            'embedding.*' => 'numeric',
            'templates' => 'nullable|array',
            'era_label' => 'nullable|string|max:100',
            'append_to_profile_id' => 'nullable|integer',
            'auto_generate_embedding' => 'nullable|boolean',
        ]);

        // Determine embedding vector (either supplied by client or simulated)
        if (!empty($validated['embedding'])) {
            $rawVector = $validated['embedding'];
            $normalizedVector = $this->recognitionService->normalizeVector($rawVector);
        } else {
            // Simulated extraction based on name seed
            $normalizedVector = $this->recognitionService->generateRealisticEmbedding($validated['name']);
        }

        $hasTemplatesCol = $this->recognitionService->ensureTemplatesColumn();

        // Support appending a new angle/era sample to an existing profile (Multi-Template per Personil)
        if (!empty($validated['append_to_profile_id'])) {
            $existing = FaceProfile::find($validated['append_to_profile_id']);
            if ($existing) {
                $templates = is_array($existing->templates) ? $existing->templates : [];
                if (empty($templates) && !empty($existing->embedding)) {
                    $templates[] = [
                        'era' => 'Sampel Awal #1',
                        'vector' => $existing->embedding,
                    ];
                }
                $newSampleNumber = count($templates) + 1;
                $templates[] = [
                    'era' => $validated['era_label'] ?? ("Sampel Sudut/Era #{$newSampleNumber}"),
                    'vector' => $normalizedVector,
                ];
                $updateData = [
                    'sample_count' => count($templates),
                    'last_seen_at' => now(),
                ];
                if ($hasTemplatesCol) {
                    $updateData['templates'] = $templates;
                }
                $existing->update($updateData);

                return response()->json([
                    'status' => 'success',
                    'message' => "Sampel wajah baru berhasil ditambahkan ke profil '{$existing->name}' (Total: " . count($templates) . " template)!",
                    'storage_info' => [
                        'photo_saved' => false,
                        'photo_reason' => 'Zero-photo architecture: Multi-template vektor disimpan tanpa foto mentah.',
                        'vector_dimensions' => count($normalizedVector),
                        'vector_storage_bytes' => count($normalizedVector) * 4 * count($templates),
                    ],
                    'data' => [
                        'id' => $existing->id,
                        'identifier' => $existing->identifier,
                        'name' => $existing->name,
                        'role' => $existing->role,
                        'sample_count' => count($templates),
                        'embedding_sample' => array_slice($normalizedVector, 0, 5),
                    ],
                ], 200);
            }
        }

        $identifier = $validated['identifier'] ?? ('ID-' . strtoupper(substr(md5(uniqid()), 0, 6)));
        $role = $validated['role'] ?? 'Member';

        // Check if identifier already exists
        if (FaceProfile::where('identifier', $identifier)->exists()) {
            return response()->json([
                'status' => 'error',
                'message' => "Identifier '{$identifier}' sudah digunakan.",
            ], 422);
        }

        $initialTemplates = !empty($validated['templates']) && is_array($validated['templates'])
            ? $validated['templates']
            : [
                [
                    'era' => $validated['era_label'] ?? 'Sampel Utama #1',
                    'vector' => $normalizedVector,
                ],
            ];

        $createData = [
            'identifier' => $identifier,
            'name' => $validated['name'],
            'role' => $role,
            'gender' => $validated['gender'] ?? 'Not Specified',
            'embedding' => $normalizedVector,
            'vector_dimensions' => count($normalizedVector),
            'sample_count' => count($initialTemplates),
            'last_seen_at' => now(),
        ];
        if ($hasTemplatesCol) {
            $createData['templates'] = $initialTemplates;
        }

        $profile = FaceProfile::create($createData);

        return response()->json([
            'status' => 'success',
            'message' => "Profil wajah '{$profile->name}' berhasil didaftarkan!",
            'storage_info' => [
                'photo_saved' => false,
                'photo_reason' => 'Zero-photo architecture: Foto awal tidak disimpan untuk efisiensi penyimpanan & privasi.',
                'vector_dimensions' => count($normalizedVector),
                'vector_storage_bytes' => count($normalizedVector) * 4,
            ],
            'data' => [
                'id' => $profile->id,
                'identifier' => $profile->identifier,
                'name' => $profile->name,
                'role' => $profile->role,
                'embedding_sample' => array_slice($normalizedVector, 0, 5),
            ],
        ], 201);
    }

    /**
     * Recognize an incoming face embedding vector (or multiple faces in a group frame).
     */
    public function recognize(Request $request): JsonResponse
    {
        if ($request->has('faces') && is_array($request->input('faces'))) {
            $validated = $request->validate([
                'faces' => 'required|array|min:1',
                'faces.*.embedding' => 'required|array|min:16',
                'faces.*.embedding.*' => 'numeric',
                'faces.*.box' => 'nullable|array',
                'threshold' => 'nullable|numeric|min:0.1|max:1.5',
                'condition_note' => 'nullable|string|max:100',
            ]);

            $threshold = $validated['threshold'] ?? FaceRecognitionService::MATCH_THRESHOLD;
            $result = $this->recognitionService->recognizeGroup(
                $validated['faces'],
                (float)$threshold,
                $validated['condition_note'] ?? 'Live Group Scan'
            );

            return response()->json($result);
        }

        $validated = $request->validate([
            'embedding' => 'required|array|min:16',
            'embedding.*' => 'numeric',
            'box' => 'nullable|array',
            'threshold' => 'nullable|numeric|min:0.1|max:1.5',
            'condition_note' => 'nullable|string|max:100',
        ]);

        $threshold = $validated['threshold'] ?? FaceRecognitionService::MATCH_THRESHOLD;
        $result = $this->recognitionService->recognize(
            $validated['embedding'],
            (float)$threshold,
            $validated['condition_note'] ?? 'Live Verification Scan'
        );
        $result['face_index'] = 1;
        $result['box'] = $validated['box'] ?? null;
        $result['total_faces_detected'] = 1;
        $result['faces'] = [$result];

        return response()->json($result);
    }

    /**
     * Import and sync curated Multi-Era Michael Jackson Biometric Model (mc/model_michael_jackson.json)
     * into the database so multi-template matching works immediately.
     */
    public function importMcModel(): JsonResponse
    {
        $mcData = $this->recognitionService->getMcModelData();
        if (!$mcData || empty($mcData['centroid_embedding'])) {
            return response()->json([
                'status' => 'error',
                'message' => 'File model mc/model_michael_jackson.json tidak ditemukan atau belum lengkap.',
            ], 404);
        }

        $centroid = $this->recognitionService->normalizeVector($mcData['centroid_embedding']);
        $templates = $mcData['templates'] ?? [];
        $hasTemplatesCol = $this->recognitionService->ensureTemplatesColumn();

        $updatePayload = [
            'embedding' => $centroid,
            'vector_dimensions' => count($centroid),
            'sample_count' => max(1, count($templates)),
            'last_seen_at' => now(),
        ];
        if ($hasTemplatesCol) {
            $updatePayload['templates'] = $templates;
        }

        // Also update any existing profile whose name contains "michael", "jackson", or "mc"
        $existingMichael = FaceProfile::where('name', 'LIKE', '%Michael%')
            ->orWhere('name', 'LIKE', '%Jackson%')
            ->orWhere('name', 'mc')
            ->orWhere('identifier', 'ID-MJ-MODEL')
            ->first();

        if ($existingMichael) {
            $existingMichael->update($updatePayload);
            $profile = $existingMichael;
        } else {
            $createPayload = array_merge([
                'name' => 'Michael Jackson',
                'role' => 'King of Pop (Multi-Era MC)',
                'gender' => 'Male',
            ], $updatePayload);

            $profile = FaceProfile::updateOrCreate(
                ['identifier' => 'ID-MJ-MODEL'],
                $createPayload
            );
        }

        return response()->json([
            'status' => 'success',
            'message' => "Model Akurat Multi-Era Michael Jackson ({$profile->sample_count} Template Era 1969–2000s) berhasil disinkronkan!",
            'data' => [
                'id' => $profile->id,
                'identifier' => $profile->identifier,
                'name' => $profile->name,
                'role' => $profile->role,
                'sample_count' => $profile->sample_count,
                'template_count' => count($templates),
                'version' => $mcData['version'] ?? '3.0.0',
            ],
        ]);
    }

    /**
     * Simulate multiple initial face registrations (e.g. 5 distinct personas).
     */
    public function seedSimulation(Request $request): JsonResponse
    {
        $defaultPersonas = [
            [
                'name' => 'Budi Santoso',
                'identifier' => 'EMP-001',
                'role' => 'Senior Engineer',
                'gender' => 'Male',
            ],
            [
                'name' => 'Siti Rahma',
                'identifier' => 'EMP-002',
                'role' => 'Product Designer',
                'gender' => 'Female',
            ],
            [
                'name' => 'Eko Prasetyo',
                'identifier' => 'EMP-003',
                'role' => 'Operations Lead',
                'gender' => 'Male',
            ],
            [
                'name' => 'Dewi Lestari',
                'identifier' => 'EMP-004',
                'role' => 'Finance Director',
                'gender' => 'Female',
            ],
            [
                'name' => 'Rian Wijaya',
                'identifier' => 'EMP-005',
                'role' => 'Security Analyst',
                'gender' => 'Male',
            ],
        ];

        $created = [];
        foreach ($defaultPersonas as $persona) {
            $embedding = $this->recognitionService->generateRealisticEmbedding($persona['name']);

            $profile = FaceProfile::updateOrCreate(
                ['identifier' => $persona['identifier']],
                [
                    'name' => $persona['name'],
                    'role' => $persona['role'],
                    'gender' => $persona['gender'],
                    'embedding' => $embedding,
                    'vector_dimensions' => count($embedding),
                    'sample_count' => 1,
                    'last_seen_at' => now()->subHours(rand(1, 48)),
                ]
            );

            $created[] = [
                'id' => $profile->id,
                'identifier' => $profile->identifier,
                'name' => $profile->name,
                'role' => $profile->role,
                'dimensions' => 128,
                'storage_bytes' => 128 * 4,
            ];
        }

        return response()->json([
            'status' => 'success',
            'message' => 'Berhasil membuat simulasi 5 profil wajah tanpa menyimpan foto awal!',
            'count' => count($created),
            'profiles' => $created,
            'storage_notice' => 'Semua profil disimpan murni dalam bentuk 128-D float vector (512 bytes per profil).',
        ]);
    }

    /**
     * Simulate a subsequent arrival/visit ("Kedatangan Berikutnya").
     */
    public function simulateArrival(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'profile_id' => 'nullable|integer',
            'condition' => 'nullable|string|in:normal,slight_angle,lighting_variation,wearing_glasses,fatigue_expression,unknown_stranger',
            'threshold' => 'nullable|numeric|min:0.1|max:1.5',
        ]);

        $condition = $validated['condition'] ?? 'normal';
        $threshold = $validated['threshold'] ?? FaceRecognitionService::MATCH_THRESHOLD;

        $targetProfile = null;
        if ($condition !== 'unknown_stranger') {
            if (!empty($validated['profile_id'])) {
                $targetProfile = FaceProfile::find($validated['profile_id']);
            } else {
                $targetProfile = FaceProfile::inRandomOrder()->first();
            }

            if (!$targetProfile) {
                return response()->json([
                    'status' => 'error',
                    'message' => 'Tidak ada profil wajah yang ditemukan. Silakan daftarkan atau jalankan simulasi seed terlebih dahulu.',
                ], 404);
            }

            $arrivalVector = $this->recognitionService->generateArrivalVariant(
                $targetProfile->embedding,
                $condition
            );
            $expectedPerson = $targetProfile->name;
        } else {
            $arrivalVector = $this->recognitionService->generateArrivalVariant(
                [],
                'unknown_stranger'
            );
            $expectedPerson = 'Orang Asing (Tidak Terdaftar)';
        }

        $conditionLabels = [
            'normal' => 'Kedatangan Normal (Kamera Depan, Pencahayaan Standar)',
            'slight_angle' => 'Wajah Agak Menoleh / Menunduk (+/- 15 derajat)',
            'lighting_variation' => 'Pencahayaan Berbeda (Cahaya Redup / Backlight)',
            'wearing_glasses' => 'Aksesoris Tambahan (Kacamata / Topi)',
            'fatigue_expression' => 'Perubahan Ekspresi Wajah (Tersenyum / Serius)',
            'unknown_stranger' => 'Pengunjung Asing (Belum pernah terdaftar sebelumnya)',
        ];

        $conditionNote = $conditionLabels[$condition] ?? 'Arrival Simulation';

        // Run recognition against the stored database
        $recognitionResult = $this->recognitionService->recognize(
            $arrivalVector,
            (float)$threshold,
            $conditionNote
        );

        return response()->json([
            'simulation_context' => [
                'expected_target' => $expectedPerson,
                'condition_key' => $condition,
                'condition_description' => $conditionNote,
                'incoming_vector_sample' => array_slice($arrivalVector, 0, 6),
                'threshold' => $threshold,
            ],
            'recognition_result' => $recognitionResult,
        ]);
    }

    /**
     * Get recognition logs.
     */
    public function getLogs(): JsonResponse
    {
        $logs = FaceRecognitionLog::with('profile')
            ->orderBy('created_at', 'desc')
            ->limit(30)
            ->get()
            ->map(function ($log) {
                return [
                    'id' => $log->id,
                    'recognized_name' => $log->recognized_name,
                    'is_match' => $log->is_match,
                    'distance' => $log->euclidean_distance,
                    'similarity' => $log->cosine_similarity,
                    'confidence' => $log->confidence_percentage,
                    'condition' => $log->condition_note,
                    'time' => $log->created_at->format('Y-m-d H:i:s'),
                    'time_ago' => $log->created_at->diffForHumans(),
                ];
            });

        return response()->json([
            'status' => 'success',
            'data' => $logs,
        ]);
    }

    /**
     * Get storage comparison analytics (Photo vs Embeddings).
     */
    public function getStorageMetrics(): JsonResponse
    {
        $metrics = $this->recognitionService->getStorageMetrics();
        return response()->json([
            'status' => 'success',
            'metrics' => $metrics,
        ]);
    }

    /**
     * Delete a single profile.
     */
    public function deleteProfile($id): JsonResponse
    {
        $profile = FaceProfile::findOrFail($id);
        $name = $profile->name;
        $profile->delete();

        return response()->json([
            'status' => 'success',
            'message' => "Profil '{$name}' berhasil dihapus.",
        ]);
    }

    /**
     * Clear all profiles and logs.
     */
    public function clearAll(): JsonResponse
    {
        Schema::disableForeignKeyConstraints();
        FaceRecognitionLog::truncate();
        FaceProfile::truncate();
        Schema::enableForeignKeyConstraints();

        return response()->json([
            'status' => 'success',
            'message' => 'Semua profil wajah dan riwayat verifikasi telah direset.',
        ]);
    }
}
