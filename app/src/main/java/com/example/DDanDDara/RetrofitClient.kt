@file:Suppress("PackageName")

package com.example.DDanDDara

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

val okHttpClient = OkHttpClient.Builder()
    .connectTimeout(60, TimeUnit.SECONDS)   // 연결 시간 60초
    .readTimeout(120, TimeUnit.SECONDS)     // 읽기 시간 120초
    .writeTimeout(120, TimeUnit.SECONDS)    // 쓰기 시간 120초
    .build()

object RetrofitClient {
    private const val BASE_URL = "http://192.168.0.100:8000/"
    //192.168.0.111
    //192.168.0.100
    //192.168.9.37
    //10.79.40.64

    val instance: ApiService by lazy {
        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()

        retrofit.create(ApiService::class.java)
    }
}