import cv2
import numpy as np
from ops.gs.basic import Frame

class Enhanced_Inpaint:
    def __init__(self, cfg, dilate_kernel_size=5, hole_min_size=10):
        self.cfg = cfg
        self.dilate_kernel_size = dilate_kernel_size
        self.hole_min_size = hole_min_size
        
    def expand_holes(self, mask, expansion_pixels=10):
        """Expand holes in the mask"""
        kernel = np.ones((expansion_pixels, expansion_pixels), np.uint8)
        expanded = cv2.dilate(mask.astype(np.uint8), kernel)
        return expanded > 0

    def find_and_expand_holes(self, frame: Frame):
        print("Initial frame.inpaint_wo_edge type:", type(frame.inpaint_wo_edge))
        print("Initial frame.inpaint_wo_edge shape:", 
            frame.inpaint_wo_edge.shape if hasattr(frame.inpaint_wo_edge, 'shape') else None)
        
        if not hasattr(frame, 'inpaint'):
            frame.inpaint = np.zeros_like(frame.dpt, dtype=bool)
        if frame.inpaint_wo_edge is None:  # Add this check
            frame.inpaint_wo_edge = np.zeros_like(frame.dpt, dtype=bool)
        
        depth_holes = (frame.dpt >= self.cfg.model.sky.value) | (frame.dpt == 0)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            depth_holes.astype(np.uint8), connectivity=8
        )
        
        additional_inpaint = np.zeros_like(depth_holes)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > self.hole_min_size:
                hole_mask = (labels == i)
                expanded_hole = self.expand_holes(hole_mask)
                additional_inpaint |= expanded_hole
        
        # Convert to numpy arrays if needed
        frame.inpaint = np.asarray(frame.inpaint, dtype=bool)
        frame.inpaint_wo_edge = np.asarray(frame.inpaint_wo_edge, dtype=bool)
        
        frame.inpaint |= additional_inpaint
        frame.inpaint_wo_edge |= additional_inpaint
        
        return frame

    def process_frame_aggressive(self, frame: Frame):
        if not hasattr(frame, 'sky') or frame.sky is None:
            frame.sky = np.zeros_like(frame.dpt, dtype=bool)
        if not hasattr(frame, 'inpaint') or frame.inpaint is None:
            frame.inpaint = np.zeros_like(frame.dpt, dtype=bool)
        if not hasattr(frame, 'inpaint_wo_edge') or frame.inpaint_wo_edge is None:
            frame.inpaint_wo_edge = np.zeros_like(frame.dpt, dtype=bool)

        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(frame.inpaint.astype(np.uint8), kernel)
        
        depth_grad_x = cv2.Sobel(frame.dpt, cv2.CV_64F, 1, 0, ksize=3)
        depth_grad_y = cv2.Sobel(frame.dpt, cv2.CV_64F, 0, 1, ksize=3)
        depth_edges = (np.abs(depth_grad_x) + np.abs(depth_grad_y)) > 0.5
        
        new_inpaint = dilated_mask > 0
        new_inpaint |= depth_edges
        new_inpaint &= ~frame.sky
        
        frame.inpaint = new_inpaint
        frame.inpaint_wo_edge = new_inpaint & ~depth_edges
        
        return frame