package com.example.hackaton

import android.content.Intent
import android.hardware.Camera
import android.media.CamcorderProfile
import android.media.MediaRecorder
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.widget.Toast
import android.widget.VideoView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.gun0912.tedpermission.PermissionListener
import com.gun0912.tedpermission.normal.TedPermission
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CameraActivity3 : AppCompatActivity(), SurfaceHolder.Callback {
    private lateinit var btnRecord: FloatingActionButton
    private lateinit var surfaceView: SurfaceView
    private lateinit var videoOverlay: VideoView
    private var camera: Camera? = null
    private var mediaRecorder: MediaRecorder? = null
    private lateinit var surfaceHolder: SurfaceHolder
    private var recording = false
    private val TAG = "CameraActivity3.kt"
    private var videoFilePath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera3)

        // 권한 요청 코드
        TedPermission.create()
            .setPermissionListener(permissionListener)
            .setRationaleMessage("녹화를 위하여 권한을 허용해주세요.")
            .setDeniedMessage("권한이 거부되었습니다. 설정 > 권한에서 허용할 수 있습니다.")
            .setPermissions(
                android.Manifest.permission.CAMERA,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                android.Manifest.permission.RECORD_AUDIO
            )
            .check()

        btnRecord = findViewById(R.id.record_btn3)
        surfaceView = findViewById(R.id.surfaceView5)
        videoOverlay = findViewById(R.id.videoOverlay3)

        // SurfaceHolder 초기화
        surfaceHolder = surfaceView.holder
        surfaceHolder.addCallback(this)
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS)

        btnRecord.setOnClickListener {
            if (recording) {
                stopRecording()
            } else {
                startRecording()
            }
        }
    }
    private fun setupVideoOverlay() {
        val videoUri = Uri.parse("android.resource://${packageName}/${R.raw.jaessbee_challenge}")

        videoOverlay.setVideoURI(videoUri)
        videoOverlay.setOnPreparedListener { mediaPlayer ->
            mediaPlayer.isLooping = false
        }

        // 동영상 재생 완료 시 녹화 중지
        videoOverlay.setOnCompletionListener {
            stopRecording()
            val intent = Intent(this, ProcessActivity::class.java).apply {
                putExtra("videoFilePath", videoFilePath)
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
            Toast.makeText(this@CameraActivity3, "녹화가 시작되었습니다.", Toast.LENGTH_SHORT).show()

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
                Toast.makeText(this@CameraActivity3, "전면 카메라를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
                return
            }
            surfaceView = findViewById(R.id.surfaceView3)
            surfaceHolder = surfaceView.holder
            surfaceHolder.addCallback(this@CameraActivity3)
            surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS)
            Toast.makeText(this@CameraActivity3, "권한 허가", Toast.LENGTH_SHORT).show()
        }

        override fun onPermissionDenied(deniedPermissions: MutableList<String>?) {
            // 권한 거부 시
            Toast.makeText(this@CameraActivity3, "권한 거부", Toast.LENGTH_SHORT).show()
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