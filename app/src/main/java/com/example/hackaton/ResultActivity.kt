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
import java.io.File
import java.io.FileInputStream

class ResultActivity : AppCompatActivity() {
    private lateinit var videoView: VideoView
    private lateinit var saveButton: Button
    private var flippedVideoPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        videoView = findViewById(R.id.video_preview)
        saveButton = findViewById(R.id.save_btn)

        flippedVideoPath = intent.getStringExtra("flippedVideoPath")
        if (flippedVideoPath != null) {
            playFlippedVideo(flippedVideoPath!!)
        } else {
            Toast.makeText(this, "반전된 영상 경로를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
        }

        saveButton.setOnClickListener {
            flippedVideoPath?.let { path ->
                saveVideoToGallery(path)
            }
        }
    }

    private fun playFlippedVideo(videoPath: String) {
        val videoUri = Uri.fromFile(File(videoPath))

        videoView.setVideoURI(videoUri)
        videoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = true
        }
        videoView.start()
    }

    private fun saveVideoToGallery(videoPath: String) {
        val videoFile = File(videoPath)
        if (!videoFile.exists()) {
            Toast.makeText(this, "비디오 파일이 존재하지 않습니다.", Toast.LENGTH_SHORT).show()
            return
        }

        val resolver = contentResolver
        val contentValues = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, "Flipped_Video_${System.currentTimeMillis()}.mp4")
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Hackaton")
        }
        resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, contentValues)?.let { galleryUri ->
            resolver.openOutputStream(galleryUri).use { outputStream ->
                FileInputStream(videoFile).use { inputStream ->
                    inputStream.copyTo(outputStream!!)
                }
            }
            Toast.makeText(this, "갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show()
        }
    }
}