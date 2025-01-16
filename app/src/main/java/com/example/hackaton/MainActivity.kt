package com.example.hackaton

import android.hardware.Camera
import android.media.AudioManager
import java.text.SimpleDateFormat
import android.media.CamcorderProfile
import android.media.MediaPlayer
import android.media.MediaRecorder
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.Button
import android.widget.Toast
import com.gun0912.tedpermission.PermissionListener
import com.gun0912.tedpermission.normal.TedPermission
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity(), SurfaceHolder.Callback {
    private lateinit var btnRecord: Button
    private lateinit var surfaceView: SurfaceView
    private var camera: Camera? = null
    private var mediaRecorder: MediaRecorder? = null
    private lateinit var surfaceHolder: SurfaceHolder
    private var recording = false
    private val url = "https://www.youtube.com/shorts/Fpmqa_ldQS0"; // your URL here
    private val TAG = "MainActivity.kt"

    private var mediaPlayer: MediaPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

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

        btnRecord = findViewById(R.id.btn_record)
        btnRecord.setOnClickListener {
            if (recording) {
                mediaPlayer?.release();
                mediaPlayer = null;

                // 녹화 중지
                mediaRecorder?.apply {
                    stop()
                    release()
                }
                camera?.lock()
                recording = false
                btnRecord.text = "녹화 시작"
            } else {
                val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())

                // 녹화 시작
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "녹화가 시작되었습니다.", Toast.LENGTH_SHORT).show()
                    try {
                        mediaRecorder = MediaRecorder().apply {
                            camera?.unlock()
                            setCamera(camera)
                            setAudioSource(MediaRecorder.AudioSource.CAMCORDER)
                            setVideoSource(MediaRecorder.VideoSource.CAMERA)

                            mediaPlayer = MediaPlayer.create(this@MainActivity, R.raw.kick_drum_base)
                            mediaPlayer?.start()

                            // 녹화 설정
                            setProfile(CamcorderProfile.get(CamcorderProfile.QUALITY_720P))
                            setOrientationHint(270)
                            setOutputFile("/storage/emulated/0/Download/video_$timeStamp.mp4")
                            setPreviewDisplay(surfaceHolder.surface)
                            prepare()
                            start()
                        }
                        recording = true
                        btnRecord.text = "녹화 종료"
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in 79: ${e.message}")
                        e.printStackTrace()
                        mediaRecorder?.release()
                    }
                }
            }
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

    private val permissionListener = object : PermissionListener {
        override fun onPermissionGranted() {
            // 권한을 허용받았을 때 카메라와 SurfaceView 설정
            val cameraId = getFrontCameraId() // 전면 카메라 ID 가져오기
            if (cameraId != -1) {
                camera = Camera.open(cameraId) // 전면 카메라 열기
                camera?.setDisplayOrientation(90)
            } else {
                Toast.makeText(this@MainActivity, "전면 카메라를 찾을 수 없습니다.", Toast.LENGTH_SHORT).show()
                return
            }
            surfaceView = findViewById(R.id.surfaceView)
            surfaceHolder = surfaceView.holder
            surfaceHolder.addCallback(this@MainActivity)
            surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS)
            Toast.makeText(this@MainActivity, "권한 허가", Toast.LENGTH_SHORT).show()
        }

        override fun onPermissionDenied(deniedPermissions: MutableList<String>?) {
            // 권한 거부 시
            Toast.makeText(this@MainActivity, "권한 거부", Toast.LENGTH_SHORT).show()
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        // Surface가 생성되었을 때 동작
        camera?.apply {
            setPreviewDisplay(holder)
            startPreview() // 미리보기 시작
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