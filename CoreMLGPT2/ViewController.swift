//
//  ViewController.swift
//  CoreMLGPT2
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    @IBOutlet weak var shuffleBtn: UIButton!
    @IBOutlet weak var triggerBtn: UIButton!
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var speedLabel: UILabel!
    
    let model = GPT2(strategy: .topK(40))
    
    let prompts = [
        "Before boarding your rocket to Mars, remember to pack these items",
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",
        "Today, scientists confirmed the worst possible outcome: the massive asteroid will collide with Earth",
        "Hugging Face is a company that releases awesome projects in machine learning because",
    ]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        shuffle()
        shuffleBtn.addTarget(self, action: #selector(shuffle), for: .touchUpInside)
        triggerBtn.addTarget(self, action: #selector(trigger), for: .touchUpInside)
        
        textView.flashScrollIndicators()
        self.speedLabel.text = "0"
    }
    
    @objc func shuffle() {
        guard let prompt = prompts.randomElement() else {
            return
        }
        textView.text = prompt
    }
    
    @objc func trigger() {
        guard let text = textView.text else {
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            _ = self.model.generate(text: text, nTokens: 50) { completion, time in
                DispatchQueue.main.async {
                    let startingTxt = NSMutableAttributedString(string: text, attributes: [
                        .font: self.textView.font as Any,
                        .foregroundColor: self.textView.textColor as Any,
                    ])
                    let completeTxt = NSAttributedString(string: completion, attributes: [
                        .font: self.textView.font as Any,
                        .foregroundColor: self.textView.textColor as Any,
                        .backgroundColor: UIColor.lightGray.withAlphaComponent(0.5),
                    ])
                    startingTxt.append(completeTxt)
                    self.textView.attributedText = startingTxt
                    self.speedLabel.text = String(format: "%.2f", 1 / time)
                }
            }
        }
    }
}
