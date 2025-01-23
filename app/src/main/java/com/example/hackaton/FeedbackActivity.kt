package com.example.hackaton

import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.Toast

class FeedbackActivity : AppCompatActivity() {

    private lateinit var frameImageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_feedback)

        frameImageView = findViewById(R.id.frameImageView)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        val frameIndex = intent.getIntExtra("frameIndex", -1)
        val frameUriString = intent.getStringExtra("frameUri")

        if (frameUriString != null) {
            val frameUri = Uri.parse(frameUriString)
            frameImageView.setImageURI(frameUri) // ImageView에 이미지 표시
        } else {
            Toast.makeText(this, "프레임 이미지를 받을 수 없습니다.", Toast.LENGTH_SHORT).show()
        }

        if (frameIndex != -1) {
            Log.d("FeedbackActivity", "프레임 인덱스: $frameIndex")
        } else {
            Toast.makeText(this, "유효하지 않은 프레임 인덱스입니다.", Toast.LENGTH_SHORT).show()
        }
    }
}