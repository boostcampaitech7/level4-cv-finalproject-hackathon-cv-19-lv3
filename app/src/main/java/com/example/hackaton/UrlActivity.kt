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
                // 다음 UrlProcessActivity로 이동
                val intent = Intent(this@UrlActivity, UrlProcessActivity::class.java).apply {
                    putExtra("youtubeUrl", youtubeUrl)
                }
                startActivity(intent)
            } else {
                Toast.makeText(this, "URL을 입력해주세요.", Toast.LENGTH_SHORT).show()
            }
        }

    }

}