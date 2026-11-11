<?php

namespace App\Console\Commands;

use App\Models\FaceProfile;
use App\Services\FaceRecognitionService;
use Illuminate\Console\Command;

class TestFaceRecognitionCommand extends Command
{
    protected $signature = 'face:test {--condition=normal : Condition for next visit}';
    protected $description = 'Simulate multi-face recognition without storing photos';

    public function handle(FaceRecognitionService $service): int
    {
        $this->info('===============================================================');
        $this->info('  FACE RECOGNITION SIMULATION: ZERO-PHOTO VECTOR ARCHITECTURE');
        $this->info('===============================================================');

        $profiles = FaceProfile::all();
        if ($profiles->isEmpty()) {
            $this->warn('No profiles found. Running seeder first...');
            $this->call('db:seed', ['--class' => 'FaceProfileSeeder']);
            $profiles = FaceProfile::all();
        }

        $this->table(
            ['ID', 'Identifier', 'Name', 'Role', 'Vector Dims', 'Storage Size', 'Photo Saved?'],
            $profiles->map(fn($p) => [
                $p->id,
                $p->identifier,
                $p->name,
                $p->role,
                $p->vector_dimensions . ' floats',
                $p->storage_size_bytes . ' Bytes (0.5 KB)',
                'NO (0 KB)',
            ])
        );

        $this->newLine();
        $condition = $this->option('condition');
        $this->info("Simulating next arrival of 'Budi Santoso' under condition: [{$condition}]...");

        $budi = FaceProfile::where('name', 'Budi Santoso')->first();
        $arrivalVector = $service->generateArrivalVariant($budi->embedding, $condition);

        $result = $service->recognize($arrivalVector, FaceRecognitionService::MATCH_THRESHOLD, "CLI Simulation [{$condition}]");

        $this->newLine();
        if ($result['is_match']) {
            $this->info(">>> STATUS: MATCH SUCCESSFUL (Wajah Dikenali)!");
            $this->line("    Identified As       : {$result['matched_profile']['name']} ({$result['matched_profile']['role']})");
            $this->line("    Euclidean Distance  : {$result['metrics']['euclidean_distance']} (Threshold: < {$result['metrics']['threshold_used']})");
            $this->line("    Cosine Similarity   : {$result['metrics']['cosine_similarity']} / 1.0000");
            $this->line("    Confidence          : {$result['metrics']['confidence_percentage']}%");
        } else {
            $this->error(">>> STATUS: NO MATCH / REJECTED!");
            $this->line("    Euclidean Distance  : {$result['metrics']['euclidean_distance']} (Threshold: < {$result['metrics']['threshold_used']})");
        }

        $this->newLine();
        $this->info('Candidate Comparison:');
        $this->table(
            ['Candidate Name', 'Distance', 'Similarity', 'Confidence', 'Verdict'],
            collect($result['candidates'])->map(fn($c) => [
                $c['name'],
                $c['distance'],
                $c['similarity'],
                $c['confidence'] . '%',
                $c['distance'] <= 0.60 ? 'MATCH' : 'DIFFERENT',
            ])
        );

        return Command::SUCCESS;
    }
}
