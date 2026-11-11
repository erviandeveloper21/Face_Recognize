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
        Schema::create('face_profiles', function (Blueprint $table) {
            $table->id();
            $table->string('identifier')->unique(); // e.g. EMP-001, USR-001
            $table->string('name');
            $table->string('role')->default('Member');
            $table->string('gender')->nullable();
            $table->json('embedding'); // 128-dimensional float vector (stored as JSON array)
            $table->integer('vector_dimensions')->default(128);
            $table->integer('sample_count')->default(1);
            $table->timestamp('last_seen_at')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('face_profiles');
    }
};
