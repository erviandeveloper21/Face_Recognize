<?php

namespace Database\Seeders;

use App\Models\FaceProfile;
use App\Services\FaceRecognitionService;
use Illuminate\Database\Seeder;

class FaceProfileSeeder extends Seeder
{
    public function run(): void
    {
        $service = app(FaceRecognitionService::class);

        $personas = [
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

        foreach ($personas as $persona) {
            $embedding = $service->generateRealisticEmbedding($persona['name']);

            FaceProfile::updateOrCreate(
                ['identifier' => $persona['identifier']],
                [
                    'name' => $persona['name'],
                    'role' => $persona['role'],
                    'gender' => $persona['gender'],
                    'embedding' => $embedding,
                    'vector_dimensions' => count($embedding),
                    'sample_count' => 1,
                    'last_seen_at' => now()->subHours(rand(2, 36)),
                ]
            );
        }
    }
}
