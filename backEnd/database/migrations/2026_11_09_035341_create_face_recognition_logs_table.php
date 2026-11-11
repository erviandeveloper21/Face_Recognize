<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('face_recognition_logs', function (Blueprint $table) {
            $table->id();
            $table->foreignId('face_profile_id')->nullable()->constrained('face_profiles')->nullOnDelete();
            $table->string('recognized_name')->nullable();
            $table->boolean('is_match')->default(false);
            $table->float('euclidean_distance', 8, 4)->nullable(); // Distance metric (lower = closer)
            $table->float('cosine_similarity', 8, 4)->nullable(); // Similarity metric (higher = closer, max 1.0)
            $table->float('confidence_percentage', 6, 2)->default(0); // e.g. 94.50%
            $table->string('condition_note')->nullable(); // e.g. "Normal Arrival", "Low Light", "Angle 15deg"
            $table->json('input_embedding_sample')->nullable(); // snippet / first few elements for debug inspection
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('face_recognition_logs');
    }
};
