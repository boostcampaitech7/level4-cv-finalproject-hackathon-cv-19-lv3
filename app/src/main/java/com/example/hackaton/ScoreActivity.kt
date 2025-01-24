package com.example.hackaton

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast

class ScoreActivity : AppCompatActivity() {

    private lateinit var feedbackBtn: Button
    private var flippedVideoPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_score)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        flippedVideoPath = intent.getStringExtra("flippedVideoPath")

        val numberRange = (0..100)
        val num = numberRange.random()

        val scoreTextView: TextView = findViewById(R.id.score_txt)
        val commentTextView: TextView = findViewById(R.id.comment_txt)
        feedbackBtn = findViewById(R.id.gotoFeedbackBtn)
        scoreTextView.text = num.toString()

        if (num < 40) {
            commentTextView.text = "동작을 하나하나 천천히 연습 후 다시 찍어볼까?"
        } else if (num >= 70) {
            commentTextView.text = "멋진데 ? 숏폼에 올려도 되겠어 !"
        } else {
            commentTextView.text = "굿! 잘했어 한번 더 연습해서 다시 찍어볼까?"
        }

        feedbackBtn.setOnClickListener {
            if (flippedVideoPath != null) {
                moveToResultActivity()
            } else {
                Toast.makeText(this, "반전된 영상 경로를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun moveToResultActivity() {
        val intent = Intent(this, ResultActivity::class.java).apply {
            putExtra("flippedVideoPath", flippedVideoPath) // 반전된 영상 경로 전달
        }
        startActivity(intent)
    }

}