"""
Exports trained SB3 PPO policy to ONNX so it can run on the Jetson Nano via
onnxruntime.
"""
 
import argparse
 
import numpy as np
import torch
from stable_baselines3 import PPO
 
IMG_SIZE = 64
 
 
class OnnxablePolicy(torch.nn.Module):
 
    def __init__(self, policy):
        super().__init__()
        assert policy.share_features_extractor, (
            "This export script assumes a shared actor/critic feature extractor "
            "(SB3 default). If you changed policy_kwargs to use separate "
            "extractors, update forward() below to call "
            "policy.mlp_extractor.forward_actor(pi_features) instead."
        )
        self.policy = policy
 
    def forward(self, image, obstacle_dist, cliff_dist):
        obs = {"image": image, "obstacle_dist": obstacle_dist, "cliff_dist": cliff_dist}
        features = self.policy.extract_features(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        action_logits = self.policy.action_net(latent_pi)
        return action_logits
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/jetbot_ppo.zip")
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
 
    model = PPO.load(args.model_path, device="cpu")
    policy = model.policy
    policy.eval()
 
    onnxable = OnnxablePolicy(policy)
    dummy_image = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)
    dummy_obstacle = torch.zeros(1, 1, dtype=torch.float32)
    dummy_cliff = torch.zeros(1, 1, dtype=torch.float32)
 
    torch.onnx.export(
        onnxable,
        (dummy_image, dummy_obstacle, dummy_cliff),
        args.output,
        input_names=["image", "obstacle_dist", "cliff_dist"],
        output_names=["action_logits"],
        opset_version=args.opset,
        dynamic_axes={
            "image": {0: "batch"},
            "obstacle_dist": {0: "batch"},
            "cliff_dist": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"Exported ONNX policy to {args.output}")
 
    _verify(args.output, onnxable, dummy_image, dummy_obstacle, dummy_cliff)
 
 
def _verify(onnx_path, torch_model, dummy_image, dummy_obstacle, dummy_cliff):
    import onnxruntime as ort
 
    with torch.no_grad():
        torch_out = torch_model(dummy_image, dummy_obstacle, dummy_cliff).numpy()
 
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        None,
        {
            "image": dummy_image.numpy(),
            "obstacle_dist": dummy_obstacle.numpy(),
            "cliff_dist": dummy_cliff.numpy(),
        },
    )[0]
 
    if not np.allclose(torch_out, onnx_out, atol=1e-4):
        raise RuntimeError("ONNX output does not match the PyTorch model")
    print("Verified: ONNX output matches PyTorch output on a dummy input.")
    print(f"Action logits shape: {onnx_out.shape} (5 discrete actions)")
 
 
if __name__ == "__main__":
    main()
