import cv2
import numpy as np
from ops.gs.basic import Frame

class Enhanced_Inpaint:
    def __init__(self, dilate_kernel_size=5, hole_min_size=100):
        self.dilate_kernel_size = dilate_kernel_size
        self.hole_min_size = hole_min_size
        
    def expand_holes(self, mask, expansion_pixels=10):
        """Expand holes in the mask"""
        kernel = np.ones((expansion_pixels, expansion_pixels), np.uint8)
        expanded = cv2.dilate(mask.astype(np.uint8), kernel)
        return expanded > 0

    def find_and_expand_holes(self, frame: Frame):
        """Find holes in depth map and expand inpainting mask around them"""
        # Create mask for depth holes (where depth is invalid or very distant)
        depth_holes = (frame.dpt >= frame.cfg.model.sky.value) | (frame.dpt == 0)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            depth_holes.astype(np.uint8), connectivity=8
        )
        
        # Create new mask for areas that need inpainting
        additional_inpaint = np.zeros_like(depth_holes)
        
        # Process each hole
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] > self.hole_min_size:
                hole_mask = (labels == i)
                # Expand this hole region
                expanded_hole = self.expand_holes(hole_mask)
                additional_inpaint |= expanded_hole
        
        # Combine with existing inpaint mask
        frame.inpaint = frame.inpaint | additional_inpaint
        # Update inpaint_wo_edge as well
        if hasattr(frame, 'inpaint_wo_edge'):
            frame.inpaint_wo_edge = frame.inpaint_wo_edge | additional_inpaint
        
        return frame

    def process_frame_aggressive(self, frame: Frame):
        """More aggressive inpainting approach"""
        # Dilate existing inpaint mask
        kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
        dilated_mask = cv2.dilate(frame.inpaint.astype(np.uint8), kernel)
        
        # Find depth discontinuities
        depth_grad_x = cv2.Sobel(frame.dpt, cv2.CV_64F, 1, 0, ksize=3)
        depth_grad_y = cv2.Sobel(frame.dpt, cv2.CV_64F, 0, 1, ksize=3)
        depth_edges = (np.abs(depth_grad_x) + np.abs(depth_grad_y)) > 0.5
        
        # Combine masks
        new_inpaint = dilated_mask > 0
        new_inpaint |= depth_edges
        
        # Don't inpaint sky
        if hasattr(frame, 'sky'):
            new_inpaint &= ~frame.sky
            
        frame.inpaint = new_inpaint
        if hasattr(frame, 'inpaint_wo_edge'):
            frame.inpaint_wo_edge = new_inpaint & ~depth_edges
            
        return frame