<?php

namespace App\Console\Commands;

use App\Models\FaceProfile;
use App\Models\FaceRecognitionLog;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Schema;

class ClearFaceDataCommand extends Command
{
    protected $signature = 'face:clear';
    protected $description = 'Clear all face profiles and recognition logs';

    public function handle(): int
    {
        $this->info('Clearing all face profiles and recognition logs...');
        
        $beforeProfiles = FaceProfile::count();
        $beforeLogs = FaceRecognitionLog::count();
        $this->info("Before clear -> Profiles: {$beforeProfiles}, Logs: {$beforeLogs}");

        Schema::disableForeignKeyConstraints();
        FaceRecognitionLog::truncate();
        FaceProfile::truncate();
        Schema::enableForeignKeyConstraints();

        $afterProfiles = FaceProfile::count();
        $afterLogs = FaceRecognitionLog::count();
        $this->info("After clear  -> Profiles: {$afterProfiles}, Logs: {$afterLogs}");
        $this->info('Success: All face profiles and logs have been cleared.');
        return Command::SUCCESS;
    }
}
