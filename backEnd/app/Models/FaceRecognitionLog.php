<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class FaceRecognitionLog extends Model
{
    use HasFactory;

    protected $fillable = [
        'face_profile_id',
        'recognized_name',
        'is_match',
        'euclidean_distance',
        'cosine_similarity',
        'confidence_percentage',
        'condition_note',
        'input_embedding_sample',
    ];

    protected $casts = [
        'is_match' => 'boolean',
        'euclidean_distance' => 'float',
        'cosine_similarity' => 'float',
        'confidence_percentage' => 'float',
        'input_embedding_sample' => 'array',
    ];

    public function profile(): BelongsTo
    {
        return $this->belongsTo(FaceProfile::class, 'face_profile_id');
    }
}
