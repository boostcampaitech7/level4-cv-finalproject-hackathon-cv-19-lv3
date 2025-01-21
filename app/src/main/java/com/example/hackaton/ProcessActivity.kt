package com.example.hackaton

import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.widget.ImageView
import com.bumptech.glide.Glide

class ProcessActivity : AppCompatActivity() {

    private lateinit var loading: ImageView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_process)

        val danceVideoUriString = intent.getStringExtra("danceVideoUri")
        val danceVideoUri = Uri.parse(danceVideoUriString)

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

        Handler().postDelayed({
            val intent = Intent(this@ProcessActivity, ResultActivity::class.java).apply{
                putExtra("danceVideoUri", danceVideoUri)
            }
            startActivity(intent)
        }, 5000)
    }
}