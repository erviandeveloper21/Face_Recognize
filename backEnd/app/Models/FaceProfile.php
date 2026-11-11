<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

class FaceProfile extends Model
{
    use HasFactory;

    protected $fillable = [
        'identifier',
        'name',
        'role',
        'gender',
        'embedding',
        'templates',
        'vector_dimensions',
        'sample_count',
        'last_seen_at',
    ];

    protected $casts = [
        'embedding' => 'array',
        'templates' => 'array',
        'vector_dimensions' => 'integer',
        'sample_count' => 'integer',
        'last_seen_at' => 'datetime',
    ];

    /**
     * Storage byte estimation for the embedding vector
     * 128 floats * 4 bytes = 512 bytes (in binary) or raw json string length.
     */
    public function getStorageSizeBytesAttribute(): int
    {
        return count($this->embedding ?? []) * 4;
    }

    public function logs(): HasMany
    {
        return $this->hasMany(FaceRecognitionLog::class);
    }
}
