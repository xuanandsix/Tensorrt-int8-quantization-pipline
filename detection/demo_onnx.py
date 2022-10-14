#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import numpy as np
import onnxruntime
import timeit

class YoloXDection:
    def __init__(self, model_path, debug = False):
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL # ORT_ENABLE_EXTENDED ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL # ORT_PARALLEL ORT_SEQUENTIAL
        self.ort_session = onnxruntime.InferenceSession(model_path, sess_options=so)  
        self.device = 'cuda' 
        
    def set_labels(self, label_list_file):
        self.class_names = []
        fptr = open(label_list_file)
        for line in fptr:
            self.class_names.append(line.strip())

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def preprocess(self, image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        dets_list = []
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = max(int(box[0]), 0)
            y0 = max(int(box[1]), 0)
            x1 = max(int(box[2]), 0)
            y1 = max(int(box[3]), 0)

            dets_list.append([x0, y0, x1, y1, score, cls_id])
            text = '{}:{: .1f}%'.format(class_names[cls_id], score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]

            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 255), thickness=1)

        return img, dets_list


    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def forward(self, origin_img):
        mean = None #(0.485, 0.456, 0.406)
        std = None #(0.229, 0.224, 0.225)
        input_shape = tuple(map(int, args.input_shape.split(',')))
        
        img, ratio = self.preprocess(origin_img, input_shape, mean, std)
        #img = img.astype(np.float16)
        
        t = timeit.default_timer()
        ort_inputs = {self.ort_session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.ort_session.run(None, ort_inputs)
        print('time:',timeit.default_timer()-t)

        predictions = self.demo_postprocess(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        dets = self.multiclass_nms(boxes_xyxy.astype(np.float32), scores, nms_thr=0.65, score_thr=0.1)
        
        dets_list = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img, dets_list = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=0.4, class_names=self.class_names)
        return origin_img, dets_list
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("ONNX Demo")
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--label_name_path', type=str, default=None, help='label_name_path')
    parser.add_argument('--input_shape', type=str, default="640,640", help='input shape')
    parser.add_argument('--image_path', type=str, default=None, help='image path')
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    args = parser.parse_args() 
    
    detection = YoloXDection(args.model_path)
    detection.set_labels(args.label_name_path)

    origin_img = cv2.imread(args.image_path)   
    output_img, dets_list = detection.forward(origin_img) 
    
    # show rect     
    if len(dets_list) > 0:
        print(dets_list)
        cv2.imwrite(args.output_path, output_img)
        print("Put rect on output.jpg")
    else:
        print('None pet detected.')

