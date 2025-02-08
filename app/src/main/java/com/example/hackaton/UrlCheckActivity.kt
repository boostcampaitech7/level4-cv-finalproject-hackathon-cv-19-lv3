package com.example.hackaton

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.SeekBar
import android.widget.Toast
import android.widget.VideoView

class UrlCheckActivity : AppCompatActivity() {

    private lateinit var videoView: VideoView
    private lateinit var playPauseBtn: Button
    private lateinit var recordBtn: Button
    private lateinit var seekBar: SeekBar
    private var youtubeVideoPath: String? = null
    private var folderId: String? = null
    private var isPlaying = false
    private val handler = android.os.Handler()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_url_check)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        youtubeVideoPath = intent.getStringExtra("originalVideoPath")
        folderId = intent.getStringExtra("folderId")

        seekBar = findViewById(R.id.videoSeekBar)

        playPauseBtn = findViewById(R.id.playPauseBtn)
        recordBtn = findViewById(R.id.recordBtn)

        videoView = findViewById(R.id.urlVideo)
        val videoUri = Uri.parse("file://$youtubeVideoPath")
        videoView.setVideoURI(videoUri)
        videoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = true
            seekBar.max = mediaPlayer.duration
            updateSeekBar()
            videoView.start()
            isPlaying = true
            playPauseBtn.text = "Pause"
        }

        playPauseBtn.setOnClickListener {
            togglePlayPause()
        }

        recordBtn.setOnClickListener {
            showPopup()
        }

        setupSeekBar()
    }
    private fun togglePlayPause() {
        if (isPlaying) {
            videoView.pause()
            playPauseBtn.text = "Play"
        } else {
            videoView.start()
            playPauseBtn.text = "Pause"
            updateSeekBar()
        }
        isPlaying = !isPlaying
    }

    private fun setupSeekBar() {
        seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (fromUser) {
                    videoView.seekTo(progress) // 사용자 입력으로 SeekBar 변경 시 동영상 위치 이동
                }
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun updateSeekBar() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                seekBar.progress = videoView.currentPosition // SeekBar를 동영상의 현재 위치로 업데이트
                if (isPlaying) {
                    updateSeekBar() // 재생 중일 경우 반복 호출
                }
            }
        }, 1000) // 1초 간격으로 업데이트
    }

    private fun showPopup() {
        val dialogView = layoutInflater.inflate(R.layout.check, null)
        val popupDialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        val confirmButton = dialogView.findViewById<Button>(R.id.confirmButton)
        confirmButton.setOnClickListener {
            Toast.makeText(this, "확인 버튼 클릭!", Toast.LENGTH_SHORT).show()
            val intent = Intent(this, CameraActivity::class.java).apply {
                putExtra("originalVideoPath", youtubeVideoPath)
                putExtra("folderId", folderId)
            }
            startActivity(intent)
        }

        popupDialog.show()
    }
}