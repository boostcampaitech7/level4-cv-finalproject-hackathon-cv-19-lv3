package com.example.hackaton

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File

class ScoreActivity : AppCompatActivity() {

    private lateinit var feedbackBtn: Button
    private var flippedVideoPath: String? = null
    private var originalVideo: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_score)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        flippedVideoPath = intent.getStringExtra("flippedVideoPath")
        originalVideo = intent.getStringExtra("originalVideo")

        val scoreTextView: TextView = findViewById(R.id.score_txt)
        val commentTextView: TextView = findViewById(R.id.comment_txt)
        feedbackBtn = findViewById(R.id.gotoFeedbackBtn)

        // 서버 연결 시 지울 코드
        val numberRange = (0..100)
        val num = numberRange.random()

        scoreTextView.text = num.toString()

        if (num < 40) {
            commentTextView.text = "동작을 하나하나 천천히 연습 후 다시 찍어볼까?"
        } else if (num >= 70) {
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
        }
        startActivity(intent)
    }

    // 서버에 영상 업로드하고 점수를 받아오는 함수
    private fun uploadVideosToServer(originalPath: String, flippedPath: String, callback: (Int) -> Unit) {
        val api = RetrofitAPI.create("http://localhost:8000")

        // 원본 비디오 파일 준비
        val originalFile = File(originalPath)
        val originalPart = MultipartBody.Part.createFormData(
            "originalVideo",
            originalFile.name,
            originalFile.asRequestBody("video/mp4".toMediaTypeOrNull())
        )

        // 반전 비디오 파일 준비
        val flippedFile = File(flippedPath)
        val flippedPart = MultipartBody.Part.createFormData(
            "recordedVideo",
            flippedFile.name,
            flippedFile.asRequestBody("video/mp4".toMediaTypeOrNull())
        )

        // API 호출
        api.uploadVideos(originalPart, flippedPart).enqueue(object : Callback<ScoreResponse> {
            override fun onResponse(call: Call<ScoreResponse>, response: Response<ScoreResponse>) {
                if (response.isSuccessful) {
                    val score = response.body()?.score ?: 0 // 서버에서 받은 점수
                    callback(score) // UI에 점수 반영
                } else {
                    Toast.makeText(this@ScoreActivity, "점수 요청에 실패했습니다.", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<ScoreResponse>, t: Throwable) {
                Toast.makeText(this@ScoreActivity, "서버와의 연결에 실패했습니다.", Toast.LENGTH_SHORT).show()
            }
        })
    }

}