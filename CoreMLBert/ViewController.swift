//
//  ViewController.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        print("===")
        let tokenizer = BertTokenizer()
        print(
            try! tokenizer.convertTokensToIds(tokens: tokenizer.tokenize(text: "My name is unaffable Jôhn."))
        )
    }
}

