from PIL import Image
import matplotlib.pyplot as plt

base = "/home/dll0202/baksh/dataset/1st-krones-vision-ai-challenge/audit_output"
# pick varied indices
samples = [
    "synth_Insect_0050.png",
    "synth_Foil___Semitransparent_0100.png",
    "synth_Paint_residue_0080.png",
    "synth_Liquid_0050.png",
    "synth_Liquid_0100.png",
    "synth_Liquid_0130.png",
]

fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4*len(samples)))
for i, fn in enumerate(samples):
    try:
        comp = Image.open(f"{base}/crops_aug/{fn}").convert("RGB")
        mask = Image.open(f"{base}/crops_aug_masks/{fn}")
        axes[i,0].imshow(comp); axes[i,0].set_title(f"composite: {fn[:30]}", fontsize=9)
        axes[i,1].imshow(mask, cmap="gray"); axes[i,1].set_title("paste mask")
        axes[i,2].imshow(comp); axes[i,2].imshow(mask, alpha=0.4, cmap="Reds")
        axes[i,2].set_title("overlay")
    except Exception as e:
        axes[i,0].set_title(f"ERROR: {e}")
    for ax in axes[i]: ax.axis("off")
plt.tight_layout(); plt.savefig("synth_mask_check2.png", dpi=80)
plt.show()