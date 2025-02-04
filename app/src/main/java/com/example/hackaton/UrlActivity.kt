package com.example.hackaton

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.EditText
import android.widget.ImageButton
import android.widget.Toast
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class UrlActivity : AppCompatActivity() {
    private lateinit var apiService: ApiService
    private lateinit var urlText: EditText
    private lateinit var urlSearchButton: ImageButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_url)

        apiService = RetrofitClient.instance

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        urlSearchButton = findViewById(R.id.urlSearchButton)
        urlText = findViewById(R.id.urlText)

        urlSearchButton.setOnClickListener {
            val youtubeUrl = urlText.text.toString()
            Log.d("Button", "Search button clicked")

            if (youtubeUrl.isNotBlank()) {
                sendUrlToServer(youtubeUrl)
            } else {
                Toast.makeText(this, "URL을 입력해주세요.", Toast.LENGTH_SHORT).show()
            }
        }

    }
    private fun sendUrlToServer(youtubeUrl: String) {
        val requestBody = mapOf("url" to youtubeUrl)
        Log.d("server", "server api")

        apiService.downloadVideo(requestBody).enqueue(object : Callback<Map<String, String>> {
            override fun onResponse(call: Call<Map<String, String>>, response: Response<Map<String, String>>) {
                Log.d("onResponse", "success")
                if (response.isSuccessful) {
                    Log.d("isSuccessful", "success")
                    val folderId = response.body()?.get("folder_id")
                    Toast.makeText(this@UrlActivity, "동영상 다운로드 요청 성공!", Toast.LENGTH_SHORT).show()
                    Log.d("folderId", "$folderId")
                    // 다음 UrlProcessActivity로 이동
                    val intent = Intent(this@UrlActivity, UrlProcessActivity::class.java).apply {
                        putExtra("folderId", folderId)
                    }
                    startActivity(intent)
                } else {
                    Toast.makeText(this@UrlActivity, "서버 오류: ${response.code()}", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onFailure(call: Call<Map<String, String>>, t: Throwable) {
                Toast.makeText(this@UrlActivity, "네트워크 오류: ${t.message}", Toast.LENGTH_SHORT).show()
            }
        })
    }
}