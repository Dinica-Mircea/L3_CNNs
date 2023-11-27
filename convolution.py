import numpy as np

def zero_pad(X, pad):
    """
    This function applies the zero padding operation on all the images in the array X
    :param X input array of images; this array has a of rank 4 (batch_size, height, width, channels)
    :param pad the amount of zeros to be added around around the spatial size of the images
    """
    # hint you might find the function numpy.pad useful for this purpose
    # keep in mind that you only need to pad the spatial dimensions (height and width)
    # TODO your code here
    # pad width
    X = np.pad(X, mode='constant', constant_values=0, pad_width=((0, 0), (pad, pad), (pad, pad), (0, 0)))
    return X

def convolution(X, W, bias, pad, stride):
    """
    This function applied to convolution operation on the input X of shape (num_samples, iH, iW, iC)
    using the filters defined by the W (filter weights) and  (bias) parameters.

    :param X - input of shape (num_samples, iH, iW, iC)
    :param W - weights, numpy array of shape (fs, fs, iC, k), where fs is the filter size,
      iC is the depth of the input volume and k is the number of filters applied on the image
    :param biases - numpy array of shape (1, 1, 1, k)
    :param pad - hyperparameter, the amount of padding to be applied
    :param stride - hyperparameter, the stride of the convolution
    """

    # 0. compute the size of the output activation map and initialize it with zeros

    num_samples = X.shape[0]
    iW = X.shape[2]
    iH = X.shape[1]
    filter_size = W.shape[0]
    k = W.shape[3]

    # TODO your code here
    # compute the output width (oW), height (oH) and number of channels (oC)
    oW = (iW - filter_size + 2 * pad) // stride + 1
    oH = (iH - filter_size + 2 * pad) // stride + 1
    oC = k
    # initialize the output activation map with zeros
    activation_map = np.zeros((num_samples,oH, oW, k))
    # end TODO your code here

    # 1. pad the samples in the input
    # TODO your code here, pad X using pad amount
    X_padded = zero_pad(X, pad)
    # end TODO your code here

    # go through each input sample
    for i in range(num_samples):
        # TODO: get the current sample from the input (use X_padded)
        X_i = X_padded[i]
        # end TODO your code here

        # loop over the spatial dimensions
        for y in range(oH):
            # TODO your code here
            # compute the current ROI in the image on which the filter will be applied (y dimension)
            # tl_y - the y coordinate of the top left corner of the current region
            # br_y - the y coordinate of the bottom right corner of the current region
            tl_y = y * stride
            br_y = tl_y+filter_size
            # end TODO your code here

            for x in range(oW):
                # TODO your code here
                # compute the current ROI in the image on which the filter will be applied (x dimension)
                # tl_x - the x coordinate of the top left corner of the current region
                # br_x - the x coordinate of the bottom right corner of the current region
                tl_x = x * stride
                br_x = tl_x + filter_size
                # end TODO your code here

                for c in range(oC):
                    # select the current ROI on which the filter will be applied
                    roi = X_padded[i, tl_y: br_y, tl_x: br_x, :]
                    w = W[:, :, :, c]
                    b = bias[:, :, :, c]

                    # TODO your code here
                    # apply the filter with the weights w and bias b on the current image roi
                    # A. compute the elemetwise product between roi and the weights of the filters (np.multiply)
                    a = np.multiply(roi, w)
                    # B. sum across all the elements of a
                    a = np.sum(a)
                    # C. add the bias term
                    a = a + b
                    # D. add the result in the appropriate position of the output activation map
                    activation_map[i, y, x, c] = a
                    # end TODO your code here
                assert (activation_map.shape == (num_samples, oH, oW, oC))
    return activation_map

