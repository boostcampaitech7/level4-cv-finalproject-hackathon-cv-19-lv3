package com.example.DDanDDara

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast

class ScoreActivity : AppCompatActivity() {

    private lateinit var feedbackBtn: Button
    private var flippedVideoPath: String? = null
    private var originalVideo: String? = null
    private var folderId: String? = null
    private var score: Int = -1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_score)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        flippedVideoPath = intent.getStringExtra("flippedVideoPath")
        originalVideo = intent.getStringExtra("originalVideo")
        folderId = intent.getStringExtra("folderId")
        score = intent.getIntExtra("score", -1)

        val scoreTextView: TextView = findViewById(R.id.score_txt)
        val commentTextView: TextView = findViewById(R.id.comment_txt)
        feedbackBtn = findViewById(R.id.gotoFeedbackBtn)

        // 서버 연결 시 지울 코드
//        val numberRange = (0..100)
//        val num = numberRange.random()

        scoreTextView.text = score.toString()

        if (score < 40) {
            commentTextView.text = "동작을 하나하나 천천히 연습 후 다시 찍어볼까?"
        } else if (score >= 70) {
            commentTextView.text = "멋진데 ? 숏폼에 올려도 되겠어 !"
        } else {
            commentTextView.text = "굿! 잘했어 한번 더 연습해서 다시 찍어볼까?"
        }

        // 서버 연결 시 코드 ( 점수 요청 )
//        if (flippedVideoPath != null && originalVideo != null) {
//            uploadVideosToServer(originalVideo!!, flippedVideoPath!!) { score ->
//                scoreTextView.text = score.toString()
//                // 점수에 따라 코멘트를 설정
//                when {
//                    score < 40 -> {
//                        commentTextView.text = "동작을 하나하나 천천히 연습 후 다시 찍어볼까?"
//                    }
//                    score >= 70 -> {
//                        commentTextView.text = "멋진데? 숏폼에 올려도 되겠어!"
//                    }
//                    else -> {
//                        commentTextView.text = "굿! 잘했어. 한 번 더 연습해서 다시 찍어볼까?"
//                    }
//                }
//            }
//        } else {
//            Toast.makeText(this, "비디오 경로를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
//        }

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
            putExtra("originalVideo", originalVideo)
            putExtra("folderId", folderId)
        }
        startActivity(intent)
    }

}