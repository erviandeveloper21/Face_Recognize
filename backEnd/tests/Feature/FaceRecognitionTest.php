<?php

namespace Tests\Feature;

use App\Models\FaceProfile;
use App\Models\FaceRecognitionLog;
use App\Services\FaceRecognitionService;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class FaceRecognitionTest extends TestCase
{
    use RefreshDatabase;

    public function test_seed_simulation_registers_multiple_faces_without_storing_photos(): void
    {
        $response = $this->postJson('/api/face/simulate-seed');

        $response->assertStatus(200)
            ->assertJsonPath('status', 'success')
            ->assertJsonPath('count', 5);

        $this->assertDatabaseCount('face_profiles', 5);

        // Verify each profile has 128 float numbers and NO photo paths
        $profile = FaceProfile::first();
        $this->assertIsArray($profile->embedding);
        $this->assertCount(128, $profile->embedding);
        $this->assertIsFloat((float)$profile->embedding[0]);
    }

    public function test_same_person_on_next_visit_is_successfully_recognized(): void
    {
        // 1. Seed profiles
        $this->postJson('/api/face/simulate-seed');
        $budi = FaceProfile::where('name', 'Budi Santoso')->first();
        $this->assertNotNull($budi);

        // 2. Simulate next arrival with slight lighting change
        $response = $this->postJson('/api/face/simulate-visit', [
            'profile_id' => $budi->id,
            'condition' => 'lighting_variation',
            'threshold' => 0.60,
        ]);

        $response->assertStatus(200);
        $result = $response->json('recognition_result');

        $this->assertTrue($result['is_match']);
        $this->assertEquals('Budi Santoso', $result['matched_profile']['name']);
        $this->assertLessThan(0.60, $result['metrics']['euclidean_distance']);
        $this->assertGreaterThan(50.0, $result['metrics']['confidence_percentage']);

        // Verify audit log exists
        $this->assertDatabaseHas('face_recognition_logs', [
            'face_profile_id' => $budi->id,
            'is_match' => true,
        ]);
    }

    public function test_unknown_stranger_is_rejected_on_arrival(): void
    {
        // 1. Seed profiles
        $this->postJson('/api/face/simulate-seed');

        // 2. Simulate unknown person arrival
        $response = $this->postJson('/api/face/simulate-visit', [
            'condition' => 'unknown_stranger',
            'threshold' => 0.60,
        ]);

        $response->assertStatus(200);
        $result = $response->json('recognition_result');

        $this->assertFalse($result['is_match']);
        $this->assertNull($result['matched_profile']);
        $this->assertGreaterThan(0.60, $result['metrics']['euclidean_distance']);
    }

    public function test_storage_metrics_calculation_proves_massive_savings(): void
    {
        $response = $this->getJson('/api/face/storage-metrics');

        $response->assertStatus(200)
            ->assertJsonPath('status', 'success')
            ->assertJsonPath('metrics.per_user.vector_binary_formatted', '512 Bytes (0.50 KB)');

        $savings = $response->json('metrics.per_user.savings_percentage');
        $this->assertGreaterThan(99.0, $savings);
    }

    public function test_can_import_curated_multi_era_mc_model_and_match_1985_group_face(): void
    {
        $importRes = $this->postJson('/api/face/import-mc-model');
        $importRes->assertStatus(200)
            ->assertJsonPath('status', 'success')
            ->assertJsonPath('data.name', 'Michael Jackson');

        $service = app(FaceRecognitionService::class);
        $mcData = $service->getMcModelData();
        $this->assertNotNull($mcData);

        // Find the 1985 template vector and simulate a noisy group-photo crop vs a non-target face
        $era1985 = collect($mcData['templates'])->first(fn($t) => str_contains($t['era'] ?? '', '1985'));
        $this->assertNotNull($era1985);

        $strangerVector = $service->generateRealisticEmbedding('cyndi_lauper_distractor');
        $michaelGroupVector = $service->generateArrivalVariant($era1985['vector'], 'normal');

        $groupRes = $this->postJson('/api/face/recognize', [
            'faces' => [
                ['embedding' => $strangerVector, 'box' => ['x' => 122, 'y' => 456, 'width' => 63, 'height' => 61]],
                ['embedding' => $michaelGroupVector, 'box' => ['x' => 255, 'y' => 382, 'width' => 35, 'height' => 51]],
            ],
            'threshold' => 0.68,
            'condition_note' => 'Group Photo Test',
        ]);

        $groupRes->assertStatus(200)
            ->assertJsonPath('target_detected', true)
            ->assertJsonPath('target_face_index', 2)
            ->assertJsonPath('highest_face_index', 2);

        $faces = $groupRes->json('faces');
        $this->assertFalse($faces[0]['is_michael']);
        $this->assertTrue($faces[1]['is_michael']);
        $this->assertGreaterThanOrEqual(84.0, $faces[1]['top_candidate']['similarity_percentage']);
    }
}
