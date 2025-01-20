package com.example.hackaton

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton

class SongActivity : AppCompatActivity() {

    private lateinit var kick_drum_base_btn: Button
    private lateinit var home_btn: ImageButton
    private lateinit var before_btn: ImageButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_song)

        kick_drum_base_btn = findViewById(R.id.kick_challenge_btn)
        home_btn = findViewById(R.id.home_btn)
        before_btn = findViewById(R.id.back_btn)

        kick_drum_base_btn.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        home_btn.setOnClickListener {
            val intent = Intent(this, StartActivity::class.java)
            startActivity(intent)
        }

        before_btn.setOnClickListener {
            val intent = Intent(this, StartActivity::class.java)
            startActivity(intent)
        }
    }
}