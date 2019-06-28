//
//  LoaderView.swift
//  SXAssetPicker
//
//  Created by Julien Chaumond on 23/06/2015.
//  Copyright (c) 2015 Hugging Face. All rights reserved.
//

import UIKit

class LoaderView: UIView {
    
    let loader = UIActivityIndicatorView(style: .whiteLarge)
    
    var isLoading = false {
        didSet {
            if isLoading {
                loader.startAnimating()
                isHidden = false
            } else {
                loader.stopAnimating()
                isHidden = true
            }
        }
    }
    
    init() {
        super.init(frame: CGRect(x: 0, y: 0, width: 80, height: 80))
        autoresizingMask = [.flexibleTopMargin, .flexibleLeftMargin, .flexibleRightMargin, .flexibleBottomMargin]
        backgroundColor = UIColor(white: 0, alpha: 0.7)
        layer.cornerRadius = 5.0
        layer.masksToBounds = true
        isHidden = true
        loader.center = center
        addSubview(loader)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func willMove(toSuperview newSuperview: UIView?) {
        // Modified behavior from other projects.
        if let newSuperview = newSuperview {
            center = newSuperview.center
        }
    }
}
