# Foundation Vision Models for Robotics: Understand, Compare, and Use

[Course home](https://github.com/zengxiong111/Foundation_of_Embodied_AI) · [简体中文](https://github.com/zengxiong111/Foundation_of_Embodied_AI/blob/main/md/Visual_State_Estimation_ZH.md) · [PDF](https://zengxiong111.github.io/zengxiong.github.io/files/Visual_State_Estimation_EN.pdf)

*Foundation of Embodied AI · Concise course notes · October 18, 2026*

> **Three questions only:** What does each family of foundation vision models actually output? How do SAM, FoundationPose, FoundationStereo, and SAM 3D differ? How should a roboticist select and combine these existing tools?
>
> **Scope:** Concepts, model selection, system composition, and limitations. Camera-calibration derivations, SE(3) filtering mathematics, policy training, code, and laboratory exercises are intentionally out of scope.

## Contents

- [1. Start with the task: what visual output does the robot need?](#vision-section-01)
- [2. Understand and distinguish the major models](#vision-section-02)
- [3. Choose and combine models for a robot task](#vision-section-03)
- [4. Practical usage: what must be checked?](#vision-section-04)

<a id="vision-section-01"></a>

## 1. Start with the task: what visual output does the robot need?

A vision model is not an interchangeable perception black box. Select the **variable that the downstream robot needs** before comparing model size, benchmark ranks, or inference speed.

| Task | Typical output | Question it answers | What it does not directly provide |
|:--|:--|:--|:--|
| Detection | Class label, bounding box, confidence score | “Where is the target region?” | Accurate contour or 3D distance |
| Instance segmentation / tracking | Pixel mask, object ID, temporal association | “Which pixels belong to this tool?” | Object depth or rigid 6D pose |
| Depth estimation | Depth map or stereo disparity | “How far is the visible surface?” | Full object geometry or an object frame |
| 3D reconstruction | Estimated mesh, shape, or texture | “What might the object look like in 3D?” | Guaranteed metric dimensions or graspability |
| 6D pose estimation | 3D translation plus 3D rotation | “Where is the tool frame and how is it oriented?” | Contact force, friction, or grasp success |

**Four distinctions are essential:** a mask is not depth; a depth map is not a complete 3D model; a shape model is not its current pose; and 6D pose is not contact state. The corresponding tools solve different problems.

**Ask four questions first:** Is locating the object enough, or is orientation required? Do you have RGB-D or a calibrated stereo pair? Do you have CAD or reference images? Is the result offline, or must it participate in real-time control?

<a id="vision-section-02"></a>

## 2. Understand and distinguish the major models

### 2.1 Core model comparison

| Model | Required inputs | Principal output | Best use and important boundary |
|:--|:--|:--|:--|
| **SAM 2** | Image/video, point/box/mask prompts | Instance masks propagated through video | **Track a specified object**; no 6D object pose. |
| **SAM 3 / 3.1** | Image/video, short text concepts, exemplars, or visual prompts | Open-vocabulary detection, masks, video tracks | **Find objects by language**; 3.1 improves multi-object video efficiency, not a universal real-time guarantee. |
| **FoundationStereo** | Calibrated, synchronized, rectified left/right images | Dense disparity convertible to metric depth | **Stereo geometry**; neither monocular depth nor object pose. |
| **Fast-FoundationStereo** | Same stereo prerequisites | Lower-latency stereo disparity | **Stereo under tighter latency budgets**; profile on the deployment GPU. |
| **FoundationPose** | Typically RGB-D, target mask, and CAD or reference images | Rigid-object 6D pose and tracking | **Initialize/update tool pose**; depends on geometry priors and good observations. |
| **SAM 3D Objects** | Object region in a single image | Candidate 3D geometry, appearance, and layout | **Shape prior without CAD**; a generated mesh is not certified engineering geometry. |
| **SAM 3D Body** | Human image, optional keypoints/masks | Human 3D body mesh and pose | **Human motion and shape**; not hand–object contact force. |

### 2.2 Three distinctions that prevent most selection errors

**SAM 2 versus SAM 3: “track a specified instance” versus “find instances matching a concept.”** If a click or bounding box already specifies the target, SAM 2 can propagate its mask through a video. If the instruction is “find every red screwdriver,” SAM 3 can use a short textual concept or example. SAM 3.1 improves efficiency when processing multiple video objects. For long, compositional instructions, a VLM or agent may still need to reduce the request to a localizable concept.

**FoundationStereo versus monocular depth: their geometric scale comes from different sources.** FoundationStereo uses a **real, calibrated stereo pair** to estimate disparity and, with camera parameters, metric depth. Monocular approaches (including selected Depth Anything 3 variants) infer geometry from images, but relative-depth checkpoints must not be confused with specialized metric-depth checkpoints. Neither should be assumed accurate enough for close-tolerance manipulation without validation. A head camera and a moving wrist camera are not automatically a valid rectified stereo pair.

**FoundationPose versus SAM 3D Objects: “where is the rigid object now?” versus “what might its shape be?”** FoundationPose uses a CAD model or reference imagery as an object prior to initialize and track pose from observations. SAM 3D Objects can produce plausible candidate geometry without existing CAD; its scale, coordinate frame, local shape, and suitability for precision tasks still need independent checks. It is not an automatic replacement for scanned, measured, or verified CAD.

### 2.3 Two useful complementary tools

- **Grounding DINO** is text-conditioned open-vocabulary object detection, primarily returning boxes. It is useful when boxes suffice or when chaining a separate segmentation model; its interface differs from the masks returned by SAM 3.
- **Depth Anything 3** supports several monocular and multi-view depth/geometry settings. The exact checkpoint matters: check whether the output is relative depth, metric depth, or multi-view geometry. It can provide a useful prior without a hardware depth sensor, but not an assumed substitute for calibrated, accurate RGB-D.

<a id="vision-section-03"></a>

## 3. Choose and combine models for a robot task

### 3.1 A minimal selection procedure

**Step 1 — Demand the minimum sufficient output.** Stop at a detection or mask if the task only needs target presence and tracking. Add depth when distance matters, and rigid 6D pose only when orientation or geometric alignment is necessary. A direct vision-action policy (VLA) may generate actions directly from images and proprioception without explicitly estimating pose.

**Step 2 — Check your sensors and object prior.** If reliable aligned RGB-D is available, use it before adding another depth estimator. If you have calibrated synchronized stereo, consider FoundationStereo or Fast-FoundationStereo. With monocular images alone, treat predicted geometry and scale cautiously. Use accurate CAD when available; otherwise consider multi-view reference images or SAM 3D Objects as a candidate shape source.

**Step 3 — Build the smallest pipeline that meets the task.** Not every foundation model needs to run sequentially on every frame. Language-based localization may run once at initialization and again on failure; lighter tracking and pose updates can handle intermediate frames.

### 3.2 Two representative robotics pipelines

**Pipeline A — Explicit geometric state for tool grasping, orientation, and assembly.**

> Language goal → **SAM 3** for localization and segmentation → **target mask + RGB-D + CAD/reference views** → **FoundationPose** for 6D pose initialization/tracking → robot controller; re-localize after tracking failure.

Choose this pipeline when the controller needs an object coordinate frame or geometric alignment errors. **SAM 3 does not itself deliver 6D pose; FoundationPose cannot conjure accurate CAD or missing depth.**

**Pipeline B — Direct visual policies for VLA and closed-loop manipulation.**

> Task language + camera images + robot proprioception → **VLA / visual policy** → robot actions; optional **SAM 3, tracker, or geometric estimator** for object monitoring, task verification, and recovery.

This pipeline need not include FoundationPose. The approaches can be hybridized: retain explicit geometry for supervision or safety checks while leaving the policy's primary action interface unchanged.

### 3.3 Four practical situations

| Robot question | Minimal suitable choice | When to extend it |
|:--|:--|:--|
| “Find the red screwdriver on the table.” | SAM 3 for a language-conditioned mask; keep tracking in video. | Add depth and pose estimation if grasp orientation matters. |
| “Pick up a tool from arbitrary orientations and rotate it to a target pose.” | RGB-D + mask + CAD/reference imagery + FoundationPose. | Re-localize after occlusion/drop; use additional feedback for contact. |
| “I have no CAD model for this unfamiliar object.” | SAM 3D Objects for a candidate mesh. | Verify scale, surface detail, and shape before precision control. |
| “I have only monocular RGB but want scene distance.” | Appropriate Depth Anything 3 variant for a geometric prior. | Add scale constraints, RGB-D, or calibrated stereo for reliable metric control. |

<a id="vision-section-04"></a>

## 4. Practical usage: what must be checked?

### 4.1 Six pre-deployment checks

1. **Input contract:** Does this specific model require RGB, RGB-D, stereo, CAD, reference images, text, boxes, or masks? Do you actually have them?
2. **Scale and frames:** Is depth in meters? Are mask and depth registered? Is the reported pose in camera or robot-base coordinates? Camera extrinsics must be calibrated separately.
3. **Time and identity:** Is the same object ID maintained across frames? What triggers re-localization after occlusion? Is the timestamp from image capture or inference completion?
4. **Failure cases:** Which component will fail on transparency, reflections, weak texture, symmetry, high-speed motion, or a hand blocking the target?
5. **Latency budget:** End-to-end delay includes sensing, inference, communication, and control. Paper-level FPS is not the robot's control rate.
6. **Licenses and interfaces:** Check official code, weights, and commercial licenses; confirm data formats, output definitions, memory demands, and supported modes.

### 4.2 Tasks that vision foundation models do not solve alone

**A confident mask does not establish an accurate pose; an accurate pose does not establish successful contact.** Dexterous grasping, tight-tolerance insertion, and tool rotation may need joint proprioception, tactile/force feedback, and corrective control. When tracking becomes unreliable or an object falls, the system should be able to pause, re-observe, and re-initialize instead of acting on stale poses.

**Evaluate the whole robot system, not only the model.** A paper demonstrates model capabilities; actual robotic reliability additionally depends on sensor data, latency, calibration, object priors, and downstream control. A visually symmetric orientation that is unobservable in the image cannot be recovered merely by increasing model size.

### 4.3 Seven-line selection cheat sheet

- **Find an object from language → SAM 3;** need only open-vocabulary boxes → Grounding DINO.
- **Track an already identified object mask → SAM 2;** examine SAM 3.1 for multiple video objects.
- **Already have reliable RGB-D → use that depth;** do not add a model without a reason.
- **Calibrated synchronized stereo for depth → FoundationStereo;** tighter latency → Fast-FoundationStereo.
- **Monocular RGB for rough geometry → Depth Anything 3;** verify the checkpoint's scale semantics.
- **Have object CAD/reference views and need rigid 6D pose → FoundationPose.**
- **Need a candidate object shape without CAD → SAM 3D Objects;** use SAM 3D Body for human 3D reconstruction.

### Official links and further reading

Capabilities and limitations above are based on the following official papers or repositories (all links are clickable):

[**SAM 2**](https://github.com/facebookresearch/sam2) · [**SAM 3 / 3.1**](https://ai.meta.com/research/sam3) · [**SAM 3.1 release**](https://ai.meta.com/blog/segment-anything-model-3/) · [**FoundationPose**](https://github.com/NVlabs/FoundationPose) · [**FoundationStereo**](https://github.com/NVlabs/FoundationStereo) · [**Fast-FoundationStereo**](https://github.com/NVlabs/Fast-FoundationStereo) · [**SAM 3D Objects**](https://github.com/facebookresearch/sam-3d-objects) · [**SAM 3D Body**](https://github.com/facebookresearch/sam-3d-body) · [**Grounding DINO**](https://github.com/IDEA-Research/GroundingDINO) · [**Depth Anything 3**](https://github.com/ByteDance-Seed/Depth-Anything-3)

**Related courses:** Camera intrinsics/extrinsics, transforms, and calibration → *Robot Camera Calibration*; demonstrations and action learning → *Imitation Learning*; visual closed-loop policies and control → *Embodied Reinforcement Learning* and *Advanced Robot Agents*. Those details are deliberately not repeated here.

*Adapted from the [original course notes](https://zengxiong111.github.io/zengxiong.github.io/files/Visual_State_Estimation_ZH.pdf) by Xiong Zeng and contributors (CC BY 4.0). Third-party licenses still apply; hardware performance has not been verified.*
