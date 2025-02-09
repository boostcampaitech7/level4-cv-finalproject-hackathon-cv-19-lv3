package com.example.DDanDDara

import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import android.widget.VideoView
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.ReturnCode
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import java.io.File

class ProcessActivity : AppCompatActivity() {

//    private lateinit var loading: ImageView
    private lateinit var videoView: VideoView
    private var videoFilePath: String? = null
    private var originalVideo: String? = null
    private var flippedVideoPath: String? = null
    private var folderId: String? = null

    private val apiService = RetrofitClient.instance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_process)

        videoFilePath = intent.getStringExtra("videoFilePath")
        Log.d("videoFilePath", "$videoFilePath")
        originalVideo = intent.getStringExtra("originalVideo")
        folderId = intent.getStringExtra("folderId")

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
//
//        loading = findViewById(R.id.gif_loading)
//        Glide.with(this)
//            .asGif()
//            .load(R.drawable.loading)
//            .into(loading)

        videoView = findViewById(R.id.clovaDubbingVideo)

        val videoUri = Uri.parse("android.resource://${packageName}/${R.raw.loading_video}")
        videoView.setVideoURI(videoUri)
        videoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = true
            videoView.start()
        }
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
                    flippedVideoPath?.let { uploadFlippedVideo(it) }
                }
            } else {
                // 처리 실패
                runOnUiThread {
                    Toast.makeText(this, "좌우 반전 실패. FFmpeg 오류 발생.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun uploadFlippedVideo(videoPath: String) {
        val videoFile = File(videoPath)
        if (!videoFile.exists()) {
            Toast.makeText(this, "비디오 파일을 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
            return
        }

        val requestBody = RequestBody.create("video/mp4".toMediaTypeOrNull(), videoFile)
        val videoPart = MultipartBody.Part.createFormData("video", videoFile.name, requestBody)
        val folderIdPart = RequestBody.create("text/plain".toMediaTypeOrNull(), folderId ?: "")

        apiService.uploadUserVideo(folderIdPart, videoPart).enqueue(object : retrofit2.Callback<Map<String, String>> {
            override fun onResponse(call: retrofit2.Call<Map<String, String>>, response: retrofit2.Response<Map<String, String>>) {
                if (response.isSuccessful) {
                    Toast.makeText(this@ProcessActivity, "비디오 업로드 성공!", Toast.LENGTH_SHORT).show()
                    fetchScore()
                } else {
                    Toast.makeText(this@ProcessActivity, "업로드 실패: ${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: retrofit2.Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@ProcessActivity, "네트워크 오류: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun fetchScore() {
        apiService.getScore(folderId).enqueue(object : retrofit2.Callback<Map<String, Int>> {
            override fun onResponse(call: retrofit2.Call<Map<String, Int>>, response: retrofit2.Response<Map<String, Int>>) {
                if (response.isSuccessful) {
                    val score = response.body()?.get("score")
                    Toast.makeText(this@ProcessActivity, "점수: $score", Toast.LENGTH_SHORT).show()
                    moveToScoreActivity(score)
                } else {
                    Toast.makeText(this@ProcessActivity, "점수 조회 실패: ${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: retrofit2.Call<Map<String, Int>>, t: Throwable) {
                Toast.makeText(this@ProcessActivity, "네트워크 오류: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun moveToScoreActivity(score: Int?) {
        val intent = Intent(this, ScoreActivity::class.java).apply {
            putExtra("flippedVideoPath", flippedVideoPath) // 반전된 영상 경로 전달
            putExtra("originalVideo", originalVideo)
            putExtra("folderId", folderId)
            putExtra("score", score)
        }
        startActivity(intent)
    }
}