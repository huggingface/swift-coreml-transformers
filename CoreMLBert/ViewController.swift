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
            tokenizer.tokenize(text: "Brave gaillard, d'où [UNK] êtes vous?")
        )
    }
}

