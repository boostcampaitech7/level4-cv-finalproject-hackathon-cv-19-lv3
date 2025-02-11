package com.example.DDanDDara

import android.content.Intent
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton

class BottomFragment : Fragment() {
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_bottom, container, false)

        // home_btn 클릭 시 StartActivity로 이동
        val homeButton: ImageButton = view.findViewById(R.id.home_btn)
        homeButton.setOnClickListener {
            val intent = Intent(requireContext(), StartActivity::class.java)
            startActivity(intent)
        }

        // back_btn 클릭 시 이전 Activity로 이동
        val backButton: ImageButton = view.findViewById(R.id.back_btn)
        backButton.setOnClickListener {
            requireActivity().onBackPressed()
        }

        return view


    }
}