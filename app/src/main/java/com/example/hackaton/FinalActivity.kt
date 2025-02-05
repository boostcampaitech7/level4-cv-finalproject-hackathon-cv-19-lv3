package com.example.hackaton

import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.Toast
import android.widget.VideoView
import java.io.File
import java.io.FileInputStream

class FinalActivity : AppCompatActivity() {

    private lateinit var originalVideoView: VideoView
    private lateinit var retryBtn: Button
    private lateinit var downloadBtn: Button
    private lateinit var mainBtn: Button
    private var flippedVideoPath: String? = null
    private var originalVideo: String? = null
    private var folderId: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_final)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        flippedVideoPath = intent.getStringExtra("flippedVideoPath")
        originalVideo = intent.getStringExtra("originalVideo")
        folderId = intent.getStringExtra("folderId")
        val videoUri = Uri.parse(originalVideo)

        retryBtn = findViewById(R.id.retryBtn)
        downloadBtn = findViewById(R.id.downloadBtn)
        mainBtn = findViewById(R.id.mainBtn)

        originalVideoView = findViewById(R.id.originalVideo)
        startVideo(videoUri)

        retryBtn.setOnClickListener {
                val intent = Intent(this, CameraActivity::class.java).apply {
                    putExtra("originalVideoPath", originalVideo)
                    putExtra("folderId", folderId)
                }
                startActivity(intent)
        }

        downloadBtn.setOnClickListener {
            flippedVideoPath?.let { path ->
                saveVideoToGallery(path)
            }
        }

        mainBtn.setOnClickListener {
            val intent = Intent(this, UrlActivity::class.java)
            startActivity(intent)
        }

    }

    private fun startVideo(videoUri: Uri) {
        originalVideoView.setVideoURI(videoUri)
        originalVideoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = true
        }

        originalVideoView.start()
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