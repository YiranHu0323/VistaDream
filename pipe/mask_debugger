import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class MaskDebugger:
    def __init__(self):
        self.debug_dir = "debug_masks"
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def visualize_masks(self, frame, step_name):
        """
        Visualize different masks and their combinations for debugging
        """
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Mask Debug - {step_name}')
        
        # Original RGB
        axes[0,0].imshow(frame.rgb)
        axes[0,0].set_title('Original RGB')
        
        # Basic inpaint mask
        if hasattr(frame, 'inpaint'):
            mask_viz = frame.inpaint.astype(np.float32)
            axes[0,1].imshow(mask_viz, cmap='gray')
            axes[0,1].set_title(f'Inpaint Mask\nCoverage: {mask_viz.mean():.2%}')
        
        # Inpaint without edge mask
        if hasattr(frame, 'inpaint_wo_edge'):
            wo_edge_viz = frame.inpaint_wo_edge.astype(np.float32)
            axes[0,2].imshow(wo_edge_viz, cmap='gray')
            axes[0,2].set_title(f'Inpaint wo Edge\nCoverage: {wo_edge_viz.mean():.2%}')
        
        # Sky mask
        if hasattr(frame, 'sky'):
            sky_viz = frame.sky.astype(np.float32)
            axes[1,0].imshow(sky_viz, cmap='gray')
            axes[1,0].set_title(f'Sky Mask\nCoverage: {sky_viz.mean():.2%}')
        
        # Depth visualization
        if hasattr(frame, 'dpt'):
            dpt_viz = frame.dpt / (frame.dpt.max() + 1e-8)
            axes[1,1].imshow(dpt_viz, cmap='viridis')
            axes[1,1].set_title('Depth Map')
        
        # Combined mask visualization
        if hasattr(frame, 'inpaint') and hasattr(frame, 'sky'):
            combined = np.zeros((*frame.inpaint.shape, 3))
            combined[frame.inpaint] = [1, 0, 0]  # Red for inpaint areas
            combined[frame.sky] = [0, 0, 1]      # Blue for sky
            if hasattr(frame, 'inpaint_wo_edge'):
                combined[frame.inpaint_wo_edge] = [0, 1, 0]  # Green for non-edge inpaint
            axes[1,2].imshow(combined)
            axes[1,2].set_title('Combined Masks\nRed: Inpaint\nBlue: Sky\nGreen: Non-edge')
        
        # Remove axes for better visualization
        for ax in axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(f'{self.debug_dir}/{step_name}.png')
        plt.close()
        
    def analyze_holes(self, frame):
        """
        Analyze holes in the reconstruction
        """
        if not hasattr(frame, 'inpaint'):
            return
            
        # Find connected components in the inpaint mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frame.inpaint.astype(np.uint8), connectivity=8
        )
        
        # Analyze each hole
        hole_stats = []
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            hole_stats.append({
                'id': i,
                'area': area,
                'aspect_ratio': width/height if height != 0 else 0,
                'position': (x + width//2, y + height//2),
                'dimensions': (width, height)
            })
            
        # Sort holes by area
        hole_stats.sort(key=lambda x: x['area'], reverse=True)
        
        return hole_stats

def debug_inpainting_pipeline(scene):
    debugger = MaskDebugger()
    
    print("Starting inpainting pipeline debug...")
    
    # Debug each frame in the scene
    for i, frame in enumerate(scene.frames):
        print(f"\nAnalyzing Frame {i}...")
        
        # Visualize masks for this frame
        debugger.visualize_masks(frame, f"frame_{i}")
        
        # Analyze holes
        hole_stats = debugger.analyze_holes(frame)
        
        if hole_stats:
            print(f"Found {len(hole_stats)} holes in frame {i}")
            print("\nTop 5 largest holes:")
            for j, hole in enumerate(hole_stats[:5]):
                print(f"Hole {j+1}:")
                print(f"  Area: {hole['area']} pixels")
                print(f"  Aspect Ratio: {hole['aspect_ratio']:.2f}")
                print(f"  Center Position: {hole['position']}")
                print(f"  Dimensions: {hole['dimensions']}")
    
    print("\nDebug visualizations saved to:", debugger.debug_dir)