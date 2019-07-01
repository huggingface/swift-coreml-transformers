//
//  ViewController.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import UIKit
import AVFoundation


class ViewController: UIViewController {
    @IBOutlet weak var shuffleBtn: UIButton!
    @IBOutlet weak var subjectField: UITextView!
    @IBOutlet weak var questionField: UITextView!
    @IBOutlet weak var answerBtn: UIButton!
    @IBOutlet weak var answerLabel: UILabel!
    let loaderView = LoaderView()
    
    let m = BertForQuestionAnswering()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.addSubview(loaderView)
        loaderView.isLoading = true
        Timer.scheduledTimer(withTimeInterval: 1, repeats: false) { _ in
            self.loaderView.isLoading = false
        }
        
        shuffle()
        shuffleBtn.addTarget(self, action: #selector(shuffle), for: .touchUpInside)
        answerBtn.addTarget(self, action: #selector(answer), for: .touchUpInside)
        
        subjectField.flashScrollIndicators()
        questionField.flashScrollIndicators()
    }
    
    @objc func shuffle() {
        answerLabel.text = ""
        guard let example = Squad.examples.randomElement() else {
            return
        }
        subjectField.text = example.context
        questionField.text = example.question
    }
    
    @objc func answer() {
        answerLabel.text = "Loading..."
        loaderView.isLoading = true
        
        let question = questionField.text ?? ""
        let context = subjectField.text ?? ""
        
        DispatchQueue.global(qos: .userInitiated).async {
            let prediction = self.m.predict(question: question, context: context)
            print("🎉", prediction)
            DispatchQueue.main.async {
                self.answerLabel.text = prediction.answer
                self.loaderView.isLoading = false
                self.say(text: prediction.answer)
            }
        }
    }
    
    func say(text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        let synthesizer = AVSpeechSynthesizer()
        synthesizer.speak(utterance)
    }
}

