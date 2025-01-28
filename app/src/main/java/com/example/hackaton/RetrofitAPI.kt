package com.example.hackaton

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import retrofit2.Call
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

// Retrofit API Interface
interface RetrofitAPI {
    // 서버로 두 개의 영상 전송
    @Multipart
    @POST("/uploadVideo")
    fun uploadVideos(
        @Part originalVideo: MultipartBody.Part, // 원본 영상
        @Part recordedVideo: MultipartBody.Part  // 촬영 영상
    ): Call<ScoreResponse>

    // 프레임 전송해 피드백 요청
    @GET("/getFeedback")
    fun getFeedback(
        @Query("frameTime") frame: Int,  // 프레임 시간 (초 단위)
        @Query("videoPath") videoPath: String  // 비디오 경로
    ): Call<FeedbackResponse>

    companion object {
        // Retrofit 인스턴스 생성
        fun create(baseUrl: String): RetrofitAPI {
            val okHttpClient = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS) // 연결 타임아웃 설정
                .writeTimeout(30, TimeUnit.SECONDS)   // 쓰기 타임아웃 설정
                .readTimeout(30, TimeUnit.SECONDS)    // 읽기 타임아웃 설정
                .build()

            val retrofit = Retrofit.Builder()
                .baseUrl(baseUrl) // 서버의 기본 URL 설정
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create()) // JSON 변환기 추가
                .build()

            return retrofit.create(RetrofitAPI::class.java)
        }
    }
}

data class ScoreResponse(
    val score: Int
)

data class FeedbackResponse(
    val feedback: String
)
