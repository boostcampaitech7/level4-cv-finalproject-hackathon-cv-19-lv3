package com.example.hackaton

import android.content.ContentValues
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.Toast
import android.widget.VideoView

class ResultActivity : AppCompatActivity() {
    private lateinit var videoView: VideoView
    private lateinit var saveButton: Button
    private var videoUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        videoView = findViewById(R.id.video_preview)
        saveButton = findViewById(R.id.save_btn)

        videoUri = intent.getParcelableExtra("danceVideoUri")
        videoUri?.let {
            Log.d("ResultActivity", "Video URI: $it")
            videoView.setVideoURI(it)
            videoView.start()
        } ?: Log.d("ResultActivity", "videoUri is null")

        videoUri?.let {
            videoView.setVideoURI(it)
            videoView.start()
        }

        saveButton.setOnClickListener {
            videoUri?.let { uri ->
                saveVideoToGallery(uri)
            }
        }


    }

    private fun saveVideoToGallery(uri: Uri) {
        val resolver = contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "RecordedVideo_${System.currentTimeMillis()}.mp4")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Hackaton")
        }
        resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)?.let { galleryUri ->
            resolver.openOutputStream(galleryUri).use { outputStream ->
                contentResolver.openInputStream(uri)?.use { inputStream ->
                    inputStream.copyTo(outputStream!!)
                }
            }
            Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
        }
    }
}