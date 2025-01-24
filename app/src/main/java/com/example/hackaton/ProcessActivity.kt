package com.example.hackaton

import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.widget.ImageView
import android.widget.Toast
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import com.bumptech.glide.Glide

class ProcessActivity : AppCompatActivity() {

    private lateinit var loading: ImageView
    private var videoFilePath: String? = null
    private var flippedVideoPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_process)

        videoFilePath = intent.getStringExtra("videoFilePath")

        if (videoFilePath != null) {
            flipVideoHorizontally(videoFilePath!!)
        } else {
            Toast.makeText(this, "영상 경로를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
        }


        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        loading = findViewById(R.id.gif_loading)
        Glide.with(this)
            .asGif()
            .load(R.drawable.loading)
            .into(loading)
    }

    private fun flipVideoHorizontally(inputPath: String) {
        flippedVideoPath = inputPath.replace(".mp4", "_flipped.mp4")
        val command = "-i $inputPath -vf hflip -c:a copy $flippedVideoPath"

        FFmpegKit.executeAsync(command) { session ->
            val state = session.state
            val returnCode = session.returnCode

            if (ReturnCode.isSuccess(returnCode)) {
                // 처리 성공
                runOnUiThread {
                    Toast.makeText(this, "좌우 반전 완료: $flippedVideoPath", Toast.LENGTH_LONG).show()
                    moveToScoreActivity()
                }
            } else {
                // 처리 실패
                runOnUiThread {
                    Toast.makeText(this, "좌우 반전 실패. FFmpeg 오류 발생.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun moveToScoreActivity() {
        val intent = Intent(this, ScoreActivity::class.java).apply {
            putExtra("flippedVideoPath", flippedVideoPath) // 반전된 영상 경로 전달
        }
        startActivity(intent)
    }
}