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
            _ = self.model.generate(text: text, nTokens: 12) { completion in
                DispatchQueue.main.async {
                    let startingTxt = NSMutableAttributedString(string: text, attributes: [
                        NSAttributedString.Key.font: self.textView.font as Any,
                    ])
                    let completeTxt = NSAttributedString(string: completion, attributes: [
                        NSAttributedString.Key.font: self.textView.font as Any,
                        NSAttributedString.Key.backgroundColor: #colorLiteral(red: 0.8257101774, green: 0.8819463849, blue: 0.9195404649, alpha: 1),
                    ])
                    startingTxt.append(completeTxt)
                    self.textView.attributedText = startingTxt
                }
            }
        }
    }
}
