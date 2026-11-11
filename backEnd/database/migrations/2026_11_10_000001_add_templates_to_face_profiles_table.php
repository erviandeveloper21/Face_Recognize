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
        if (!Schema::hasColumn('face_profiles', 'templates')) {
            Schema::table('face_profiles', function (Blueprint $table) {
                $table->json('templates')->nullable()->after('embedding')->comment('Multi-era / multi-angle 128-D face templates');
            });
        }
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        if (Schema::hasColumn('face_profiles', 'templates')) {
            Schema::table('face_profiles', function (Blueprint $table) {
                $table->dropColumn('templates');
            });
        }
    }
};
