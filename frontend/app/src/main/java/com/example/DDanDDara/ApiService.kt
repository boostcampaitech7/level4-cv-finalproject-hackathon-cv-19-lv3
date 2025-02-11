package com.example.DDanDDara

import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.DELETE
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path

data class FeedbackRequest(
    val folderId: String,
    val frame: String
)

interface ApiService {
    @POST("/video/url")
    fun downloadVideo(@Body requestBody: Map<String, String?>): Call<Map<String, String>>

    @GET("/video/{folder_id}")
    fun getVideoById(@Path("folder_id") folderId: String?): Call<ResponseBody>

    @GET("/pose/points/{folder_id}")
    fun extractVideoPose(@Path("folder_id") folderId: String?): Call<Map<String, String>>

    @Multipart
    @POST("/pose/user")
    fun uploadUserVideo(
        @Part("folder_id") id: RequestBody,
        @Part video: MultipartBody.Part
    ): Call<Map<String, String>>

    @GET("/score/{folder_id}")
    fun getScore(@Path("folder_id") folderId: String?): Call<Map<String, Int>>

    @POST("/feedback")
    fun getFeedback(@Body request: FeedbackRequest): Call<Map<String, String>>

    @DELETE("/feedback/{folder_id}")
    fun clearCacheAndFiles(@Path("folder_id") folderId: String?): Call<Map<String, String>>
}