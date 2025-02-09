package com.example.DDanDDara

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class SongActivity : AppCompatActivity() {

    private lateinit var kick_drum_base_btn: Button
    private lateinit var sticky_btn: Button
    private lateinit var jaessbee_btn: Button
    private lateinit var wait_btn: Button
    private lateinit var imok_btn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_song)

        supportFragmentManager.beginTransaction()
            .replace(R.id.topFragmentContainer, TopFragment())
            .commit()

        supportFragmentManager.beginTransaction()
            .replace(R.id.bottomFragmentContainer, BottomFragment())
            .commit()

        kick_drum_base_btn = findViewById(R.id.kick_challenge_btn)
        sticky_btn = findViewById(R.id.sticky_challenge_btn)
        jaessbee_btn = findViewById(R.id.jaessbee_challenge_btn)
        wait_btn = findViewById(R.id.wait_challenge_btn)
        imok_btn = findViewById(R.id.imok_challenge_btn)

        kick_drum_base_btn.setOnClickListener {
            val intent = Intent(this, CameraActivity1::class.java)
            startActivity(intent)
        }

        sticky_btn.setOnClickListener {
            val intent = Intent(this, CameraActivity2::class.java)
            startActivity(intent)
        }

        jaessbee_btn.setOnClickListener {
            val intent = Intent(this, CameraActivity3::class.java)
            startActivity(intent)
        }

        wait_btn.setOnClickListener {
            val intent = Intent(this, CameraActivity4::class.java)
            startActivity(intent)
        }

        imok_btn.setOnClickListener {
            val intent = Intent(this, CameraActivity5::class.java)
            startActivity(intent)
        }
    }
}