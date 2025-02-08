package com.example.hackaton

import android.content.Intent
import android.hardware.Camera
import android.media.CamcorderProfile
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.widget.Toast
import android.widget.VideoView
import androidx.core.content.FileProvider
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.gun0912.tedpermission.PermissionListener
import com.gun0912.tedpermission.normal.TedPermission
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CameraActivity : AppCompatActivity(), SurfaceHolder.Callback {

    private val REQUEST_MANAGE_STORAGE_PERMISSION = 102

    private lateinit var btnRecord: FloatingActionButton
    private lateinit var surfaceView: SurfaceView
    private lateinit var countdownVideo: VideoView
    private lateinit var videoOverlay: VideoView
    private var camera: Camera? = null
    private var mediaRecorder: MediaRecorder? = null
    private lateinit var surfaceHolder: SurfaceHolder
    private var recording = false
    private val TAG = "CameraActivity.kt"
    private var youtubeVideoPath: String? = null
    private var videoFilePath: String? = null
    private var folderId: String? = null

    private val apiService = RetrofitClient.instance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        // MANAGE_EXTERNAL_STORAGE 권한 체크 및 요청
        if (!hasManageExternalStoragePermission()) {
            requestManageExternalStoragePermission()
        }

        youtubeVideoPath = intent.getStringExtra("originalVideoPath")
        folderId = intent.getStringExtra("folderId")
        Log.d(TAG, "$folderId")


        // 권한 요청 코드
        TedPermission.create()
            .setPermissionListener(permissionListener)
            .setRationaleMessage("녹화를 위하여 권한을 허용해주세요.")
            .setDeniedMessage("권한이 거부되었습니다. 설정 > 권한에서 허용할 수 있습니다.")
            .setPermissions(
                android.Manifest.permission.CAMERA,
//                android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
//                android.Manifest.permission.READ_EXTERNAL_STORAGE,
                android.Manifest.permission.RECORD_AUDIO
            )
            .check()

        btnRecord = findViewById(R.id.record_btn)
        surfaceView = findViewById(R.id.surfaceView)
        videoOverlay = findViewById(R.id.videoOverlay)
        countdownVideo = findViewById(R.id.countdownVideo)

        // SurfaceHolder 초기화
        surfaceHolder = surfaceView.holder
        surfaceHolder.addCallback(this)
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS)

        btnRecord.setOnClickListener {
            if (recording) {
                stopRecording()
            } else {
                showCountdown()
            }
        }

        // CameraActivity 시작 시 바로 비디오 포즈 추출 API 호출
        folderId?.let {
            extractVideoPose(folderId!!)
        }
    }

    // 비디오 포즈 추출 API를 백그라운드에서 실행
    private fun extractVideoPose(folderId: String) {
        // 백그라운드에서 API 호출 처리
        Thread {
            try {
                val response = apiService.extractVideoPose(folderId).execute()
                if (response.isSuccessful) {
                    val message = response.body()?.get("message")
                    Log.d(TAG, "포즈 추출 완료: $message")

                    runOnUiThread {
                        Toast.makeText(this, "포즈 추출 작업 완료", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    Log.e(TAG, "포즈 추출 실패: ${response.code()}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "포즈 추출 오류: ${e.message}")
            }
        }.start()
    }

    private fun showCountdown() {
        val videoUri = Uri.parse("android.resource://${packageName}/${R.raw.countdown1}")

        countdownVideo.setVideoURI(videoUri)
        countdownVideo.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = false
        }

        // 동영상 재생 완료 시 녹화 중지
        countdownVideo.setOnCompletionListener {
            // 동영상 오버레이 숨김 처리
            countdownVideo.visibility = View.INVISIBLE
            startRecording()
        }

        countdownVideo.visibility = View.VISIBLE // 비디오 오버레이를 표시
        countdownVideo.start()
    }

    private fun setupVideoOverlay() {
        val videoUri = Uri.parse("file://$youtubeVideoPath")

        videoOverlay.setVideoURI(videoUri)
        videoOverlay.setOnPreparedListener { mediaPlayer ->
            Log.d("VideoView", "Video is prepared")
            mediaPlayer.isLooping = false
        }

        // 동영상 재생 완료 시 녹화 중지
        videoOverlay.setOnCompletionListener {
            stopRecording()
            val intent = Intent(this, ProcessActivity::class.java).apply {
                putExtra("videoFilePath", videoFilePath)
                putExtra("originalVideo", videoUri.toString())
                putExtra("folderId", folderId)
            }
            startActivity(intent)
        }

        videoOverlay.visibility = View.VISIBLE // 비디오 오버레이를 표시
        videoOverlay.start()
    }

    private fun startRecording() {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val videoFile = File(
            File("${filesDir}/video").apply {
                if (!this.exists()) {
                    this.mkdirs()
                }
            },
            "video_$timeStamp.mp4"
        )

        videoFilePath = videoFile.absolutePath

        // 동영상 오버레이 설정
        setupVideoOverlay()

        // 녹화 시작
        runOnUiThread {
            Toast.makeText(this@CameraActivity, "녹화가 시작되었습니다.", Toast.LENGTH_SHORT).show()

            try {
                mediaRecorder = MediaRecorder().apply {
                    camera?.unlock()
                    setCamera(camera)
                    setAudioSource(MediaRecorder.AudioSource.CAMCORDER)
                    setVideoSource(MediaRecorder.VideoSource.CAMERA)

                    // 녹화 설정
                    setProfile(CamcorderProfile.get(CamcorderProfile.QUALITY_720P))
                    setOrientationHint(270)
                    setOutputFile(videoFile.absoluteFile)
                    setPreviewDisplay(surfaceHolder.surface)
                    prepare()
                    start()
                }

                recording = true

            } catch (e: Exception) {
                Log.e(TAG, "Error in startRecording: ${e.message}")
                e.printStackTrace()
                mediaRecorder?.release()
                recording = false
            }
        }
    }
    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            camera?.lock()
            mediaRecorder = null
            recording = false
            Toast.makeText(this, "녹화가 종료되었습니다.", Toast.LENGTH_SHORT).show()
            // 동영상 오버레이 숨김 처리
            videoOverlay.visibility = View.INVISIBLE

        } catch (e: Exception) {
            Log.e(TAG, "녹화 중지 오류: ${e.message}")
            e.printStackTrace()
        }
    }

    // MANAGE_EXTERNAL_STORAGE 권한 체크
    private fun hasManageExternalStoragePermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            Environment.isExternalStorageManager()
        } else {
            true
        }
    }

    // MANAGE_EXTERNAL_STORAGE 권한 요청
    private fun requestManageExternalStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                data = Uri.parse("package:$packageName")
            }
            startActivityForResult(intent, REQUEST_MANAGE_STORAGE_PERMISSION)
        }
    }

    // 권한 요청 결과 처리
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_MANAGE_STORAGE_PERMISSION) {
            if (hasManageExternalStoragePermission()) {
                Toast.makeText(this, "MANAGE_EXTERNAL_STORAGE 권한 허용됨", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "MANAGE_EXTERNAL_STORAGE 권한 거부됨", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val permissionListener = object : PermissionListener {
        override fun onPermissionGranted() {
            // 권한을 허용받았을 때 카메라와 SurfaceView 설정
            val cameraId = getFrontCameraId() // 전면 카메라 ID 가져오기
            if (cameraId != -1) {
                camera = Camera.open(cameraId) // 전면 카메라 열기
                camera?.setDisplayOrientation(90)
                try {
                    camera?.setPreviewDisplay(surfaceHolder)
                    camera?.startPreview() // 카메라 미리보기 시작
                } catch (e: Exception) {
                    Log.e(TAG, "Error setting camera preview: ${e.message}")
                }
            } else {
                Toast.makeText(this@CameraActivity, "전면 카메라를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
                return
            }
            surfaceView = findViewById(R.id.surfaceView)
            surfaceHolder = surfaceView.holder
            surfaceHolder.addCallback(this@CameraActivity)
            surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS)
            Toast.makeText(this@CameraActivity, "권한 허가", Toast.LENGTH_SHORT).show()
        }

        override fun onPermissionDenied(deniedPermissions: MutableList<String>?) {
            // 권한 거부 시
            Toast.makeText(this@CameraActivity, "권한 거부", Toast.LENGTH_SHORT).show()
        }
    }

    // 전면 카메라 ID 가져오기 함수
    private fun getFrontCameraId(): Int {
        val numberOfCameras = Camera.getNumberOfCameras()
        val cameraInfo = Camera.CameraInfo()
        for (i in 0 until numberOfCameras) {
            Camera.getCameraInfo(i, cameraInfo)
            if (cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                return i // 전면 카메라 ID 반환
            }
        }
        return -1 // 전면 카메라를 찾지 못함
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        // Surface가 생성되었을 때 동작
        camera?.apply {
            try {
                setPreviewDisplay(holder)
                startPreview() // 수정: Surface가 생성되면 카메라 프리뷰 시작
            } catch (e: Exception) {
                Log.e(TAG, "Error setting camera preview in surfaceCreated: ${e.message}")
            }
        }
    }

    private fun refreshCamera(camera: Camera?) {
        if (surfaceHolder.surface == null) {
            return
        }
        try {
            camera?.stopPreview()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        setCamera(camera)
    }

    private fun setCamera(cam: Camera?) {
        camera = cam
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        refreshCamera(camera)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        // Surface가 파괴되었을 때 동작
    }
}