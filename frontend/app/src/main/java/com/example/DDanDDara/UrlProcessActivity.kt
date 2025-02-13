package com.example.DDanDDara

import android.content.Context
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.ImageView
import android.widget.Toast
import com.bumptech.glide.Glide
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class UrlProcessActivity : AppCompatActivity() {
    private lateinit var loading: ImageView
    private val TAG = "UrlProcessActivity"
    private lateinit var apiService: ApiService
    private var youtubeUrl: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_url_process)

        apiService = RetrofitClient.instance

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

        youtubeUrl = intent.getStringExtra("youtubeUrl")

        sendUrlToServer(youtubeUrl!!)


    }

//    private fun sendUrlToServer(youtubeUrl: String?) {
//        val requestBody = mapOf("url" to youtubeUrl)
//
//        apiService.downloadVideo(requestBody).enqueue(object : Callback<Map<String, String>> {
//            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
//                if (response.isSuccessful) {
//                    Toast.makeText(this@UrlProcessActivity, "동영상 다운로드 성공!", Toast.LENGTH_SHORT).show()
//
//                    folderId?.let {
//                        // 다운로드된 비디오 경로를 서버에서 받아오기
//                        val videoPath = response.body()?.get("videoPath") ?: return
//                        // 비디오를 파일로 저장하고 CameraActivity로 넘기기
//                        getVideoById(folderId!!)
//                    }
//                } else {
//                    Log.e(TAG, "서버 오류: ${response.code()}")
//                }
//            }
//
//            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
//                Log.e(TAG, "네트워크 오류: ${t.message}")
//            }
//        })
//    }
    private fun sendUrlToServer(youtubeUrl: String) {
        val requestBody = mapOf("url" to youtubeUrl)

        apiService.downloadVideo(requestBody).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                Log.d("onResponse", "success")
                if (response.isSuccessful) {
                    Log.d("isSuccessful", "success")
                    val folderId = response.body()?.get("folder_id")
                    Toast.makeText(this@UrlProcessActivity, "동영상 다운로드 요청 성공!", Toast.LENGTH_SHORT).show()

                    getVideoById(apiService, folderId!!, this@UrlProcessActivity) { resultMessage, videoUri ->
                        if (videoUri != null) {
                            Log.d(TAG, "비디오 스트리밍 성공: $videoUri")

                        } else {
                            Log.e(TAG, resultMessage)
                            Toast.makeText(this@UrlProcessActivity, resultMessage, Toast.LENGTH_SHORT).show()
                        }
                    }

                } else {
                    Toast.makeText(this@UrlProcessActivity, "서버 오류: ${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@UrlProcessActivity, "네트워크 오류: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }

    private fun getVideoById(
        apiService: ApiService,
        folderId: String?,
        context: Context,
        onResult: (String, String?) -> Unit
    ) {
        apiService.getVideoById(folderId).enqueue(object : Callback<ResponseBody> {
            override fun onResponse(call: Call<ResponseBody>, response: Response<ResponseBody>) {
                if (response.isSuccessful && response.body() != null) {
                    response.body()?.let { body ->
                        saveVideoToFile(body, folderId)
                    }
                } else {
                    val errorMessage = "응답 실패: ${response.code()}"
                    onResult(errorMessage, null)
                }
            }

            override fun onFailure(call: Call<ResponseBody>, t: Throwable) {
                onResult("네트워크 오류: ${t.localizedMessage}", null)
            }
        })
    }

    private fun saveVideoToFile(responseBody: ResponseBody, folderId: String?) {
        try {
            val videoFile = File(getExternalFilesDir(Environment.DIRECTORY_MOVIES), "${folderId}.mp4")
            val inputStream: InputStream = responseBody?.byteStream() ?: return
            val outputStream = FileOutputStream(videoFile)

            val buffer = ByteArray(4096)
            var bytesRead: Int

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }

            outputStream.close()
            inputStream.close()

            Log.d(TAG, "Video saved at: ${videoFile.absolutePath}")

            // 다음 UrlCheckActivity로 이동
            val intent = Intent(this, UrlCheckActivity::class.java).apply {
                putExtra("originalVideoPath", videoFile.absolutePath)
                putExtra("folderId", folderId)
            }
            startActivity(intent)

        } catch (e: Exception) {
            Log.e(TAG, "Error saving video: ${e.message}")
        }
    }
}