<?php

use App\Http\Controllers\FaceRecognitionController;
use Illuminate\Support\Facades\Route;

Route::get('/ping', function () {
    return response()->json(['status' => 'ok', 'timestamp' => now()]);
});

Route::prefix('face')->group(function () {
    // Profile Management
    Route::get('/profiles', [FaceRecognitionController::class, 'getProfiles']);
    Route::post('/register', [FaceRecognitionController::class, 'register']);
    Route::post('/import-mc-model', [FaceRecognitionController::class, 'importMcModel']);
    Route::delete('/profiles/{id}', [FaceRecognitionController::class, 'deleteProfile']);
    Route::delete('/clear', [FaceRecognitionController::class, 'clearAll']);

    // Recognition & Verification
    Route::post('/recognize', [FaceRecognitionController::class, 'recognize']);

    // Simulation Suite
    Route::post('/simulate-seed', [FaceRecognitionController::class, 'seedSimulation']);
    Route::post('/simulate-visit', [FaceRecognitionController::class, 'simulateArrival']);

    // Analytics & Logs
    Route::get('/logs', [FaceRecognitionController::class, 'getLogs']);
    Route::get('/storage-metrics', [FaceRecognitionController::class, 'getStorageMetrics']);
});
