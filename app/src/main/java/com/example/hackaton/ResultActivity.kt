package com.example.hackaton

import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.io.FileOutputStream

class ResultActivity : AppCompatActivity() {
    private lateinit var videoView: VideoView
    private lateinit var finishFeedbackButton: Button
    private lateinit var playPauseButton: Button
    private lateinit var feedbackButton: Button
    private lateinit var seekBar: SeekBar
    private lateinit var frameImageView: ImageView
    private lateinit var feedbackTextView: TextView
    private var flippedVideoPath: String? = null
    private var originalVideo: String? = null
    private var folderId: String? = null
    private var isPlaying = false
    private val handler = android.os.Handler()

    private val apiService = RetrofitClient.instance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)
        videoView = findViewById(R.id.video_preview)
        finishFeedbackButton = findViewById(R.id.finishFeedbackBtn)
        playPauseButton = findViewById(R.id.play_pause_btn)
        feedbackButton = findViewById(R.id.feedback_btn)
        seekBar = findViewById(R.id.video_seek_bar)
        frameImageView = findViewById(R.id.frameImageView)
        feedbackTextView = findViewById(R.id.feedbackTxt)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        flippedVideoPath = intent.getStringExtra("flippedVideoPath")
        originalVideo = intent.getStringExtra("originalVideo")
        folderId = intent.getStringExtra("folderId")

        if (flippedVideoPath != null) {
            playFlippedVideo(flippedVideoPath!!)
        } else {
            Toast.makeText(this, "반전된 영상 경로를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
        }

        finishFeedbackButton.setOnClickListener {
            val intent = Intent(this, FinalActivity::class.java).apply {
                putExtra("flippedVideoPath", flippedVideoPath)
                putExtra("originalVideo", originalVideo)
                putExtra("folderId", folderId)
            }
            startActivity(intent)
        }

        playPauseButton.setOnClickListener {
            togglePlayPause()
        }

        feedbackButton.setOnClickListener {
            flippedVideoPath?.let { flippedPath ->
                originalVideo?.let { originalPath ->
                    val currentPosition = videoView.currentPosition // 현재 재생 위치 (밀리초 단위)
                    val frameIndex = (currentPosition / (1000 / 30))
                    Log.d("currentFrameIndex", "$frameIndex")
                    // 서버 피드백 요청
                    feedbackRequest(frameIndex, originalPath)
                }
            }
        }

        setupSeekBar()
    }

//    private fun frameOutput(videoPath: String, frameTimeMs: Int): Int {
//        val retriever = MediaMetadataRetriever()
//        try {
//            retriever.setDataSource(videoPath) // 비디오 파일 경로 설정
//            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
//            val durationMs = durationStr?.toIntOrNull() ?: 0
//
//            if (frameTimeMs > durationMs) {
//                Log.e("FrameOutput", "지정된 시간이 동영상 길이를 초과했습니다.")
//                return -1 // 잘못된 인덱스
//            }
//
//            val frameBitmap = retriever.getFrameAtTime(frameTimeMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
//
//            if (frameBitmap != null) {
//                saveFrameToCache(frameBitmap) // 프레임 이미지를 임시 저장 (원한다면)
//                return frameTimeMs // 성공적으로 캡처된 프레임의 시간 반환
//            } else {
//                Log.e("FrameOutput", "프레임을 가져올 수 없습니다.")
//                return -1
//            }
//        } catch (e: Exception) {
//            e.printStackTrace()
//            return -1
//        } finally {
//            retriever.release() // 리소스 해제
//        }
//    }

    private fun saveFrameToCache(bitmap: Bitmap): Uri {
        val cachePath = File(cacheDir, "frames")
        if (!cachePath.exists()) {
            cachePath.mkdirs()
        }
        val file = File(cachePath, "frame_${System.currentTimeMillis()}.png")
        FileOutputStream(file).use { fos ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }
        return Uri.fromFile(file)
    }

    private fun playFlippedVideo(videoPath: String) {
        val videoUri = Uri.fromFile(File(videoPath))

        videoView.setVideoURI(videoUri)
        videoView.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = true
            seekBar.max = mediaPlayer.duration
            updateSeekBar()
        }
        videoView.start()
        isPlaying = true
        playPauseButton.text = "Pause"
    }

    private fun togglePlayPause() {
        if (isPlaying) {
            videoView.pause()
            playPauseButton.text = "Play"
        } else {
            videoView.start()
            playPauseButton.text = "Pause"
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

    private fun feedbackRequest(frame: Int, videoPath: String) {
        if (folderId.isNullOrEmpty()) {
            Toast.makeText(this, "폴더 ID가 누락되었습니다.", Toast.LENGTH_SHORT).show()
            return
        }

        val request = FeedbackRequest(
            folderId = folderId!!,
            frame = frame.toString()
        )

        apiService.getFeedback(request).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                if (response.isSuccessful) {
                    Log.d("Response Body", "${response.body()}")
                    val feedback = response.body()?.get("feedback")
                    val targetFrame = response.body()?.get("frame")?.toIntOrNull()
                    Log.d("frame", "$targetFrame")
                    if (targetFrame != null) {
                        val frameBitmap = extractFrameAtTime(targetFrame * 1000L * 30, videoPath)
                        Log.d("Bitmap", "$frameBitmap")
                        frameBitmap?.let { bitmap ->
                            val frameUri = saveFrameToCache(bitmap)
                            findViewById<FrameLayout>(R.id.feedbackFrameLayout).visibility = View.VISIBLE
                            frameImageView.setImageURI(frameUri)
                        } ?: run {
                            Toast.makeText(
                                this@ResultActivity,
                                "프레임 추출 실패.",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    } else {
                        Toast.makeText(this@ResultActivity, "올바른 프레임 값을 받지 못했습니다.", Toast.LENGTH_SHORT).show()
                    }
                    if (!feedback.isNullOrEmpty()) {
                        feedbackTextView.text = feedback // 피드백 텍스트를 화면에 표시
                    } else {
                        Toast.makeText(this@ResultActivity, "피드백을 받을 수 없습니다.", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    Toast.makeText(this@ResultActivity, "서버 응답 오류", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@ResultActivity, "서버 연결 실패: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun extractFrameAtTime(frameTimeMs: Long, videoPath: String): Bitmap? {
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(videoPath)
            retriever.getFrameAtTime(frameTimeMs, MediaMetadataRetriever.OPTION_CLOSEST)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        } finally {
            retriever.release()
        }
    }
}