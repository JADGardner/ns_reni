RE: TPAMI-2023-12-2629.R1, "RENI++: A Rotation-Equivariant, Scale-Invariant, Natural Illumination Prior"
Manuscript Type: Regular (S1)

Dear Mr. James Gardner,

Your Major Revision is due in 1 month. Please contact us as soon as possible if you need an extension.

When you are ready to submit your revision, visit the following link:
*** PLEASE NOTE: This is a two-step process. After clicking on the link, you will be directed to a webpage to confirm. ***

https://mc.manuscriptcentral.com/tpami-cs?URL_MASK=2555827a52d241339e51392ecc151afe

Once the revised manuscript is prepared, you can upload it and submit it through your Author Center.

When submitting your revised manuscript, you will be able to respond to the comments made by the reviewer(s) in the space provided. You can use this space to document any changes you make to the original manuscript. In order to expedite the processing of the revised manuscript, please be as specific as possible in your response to the reviewer(s)’ questions and comments. You may also upload your responses as separate files for review along with your revision. If you choose to do this, please choose “Summary of Changes” as the file designation.

IMPORTANT: Your original files are available to you when you upload your revised manuscript. Please delete any redundant files before completing the submission.

When the submission process is complete, you will receive an automated confirmation email immediately. If you did not receive that email, your submission is not yet complete.  

I will contact you should we have any concerns or questions regarding your revision. Otherwise, your revision will be forwarded to the assigned Associate Editor for further evaluation and processing.

Please be mindful when making your revisions that you still need to maintain the size limitations for papers submitted to TPAMI. Our manuscript types and submission length guidelines (including the main text, the abstract, index terms, illustrations and references) are found at,

http://www.computer.org/portal/web/peerreviewjournals/author#manuscript

Please note that double column will translate more readily into the final publication format.  Our peer review double column templates can be found at,

http://www.computer.org/portal/web/peerreviewjournals/author#templates

Please do not hesitate in contacting us should you have any questions about our process or are experiencing technical difficulties. You can reach me at jarnold@computer.org.

Thank you for your contribution to TPAMI, and we look forward to receiving your revised manuscript.

Thank you,

Mrs. Joyce Arnold
Administrator
IEEE Transactions on Pattern Analysis and Machine Intelligence
jarnold@computer.org

**************
Editor Comments

Associate Editor
Comments to the Author:
This revision receives an 'accept', a 'major revision', and a 'minor revision'. The reviewer recommending 'major revision' considers the novelty as limited and urges the authors to compare with the relevant works identified in the first round of review. That reviewer also questions some discussions in the response letter. The most positive reviewer also commended that the authors did not address the questions from R1 in the first round of review and therefore the paper is not ready for publication. Another newly invited reviewer also believes this work is not quite ready for publication for many missing comparisons. Overall, the AE feels this work needs a much thorough revision and suggests the authors to address ALL the questions raised by the reviewers.

********************

Reviewer Comments

Please note that some reviewers may have included additional comments in a separate file. If a review contains the note "see the attached file" under Section III A - Public Comments, you will need to log on to the submission site to view the file. After logging in to the submission site, select the Author Center. Then, click on "Submitted Manuscripts," find the correct paper, and click on "View Decision Letter." Scroll down to the bottom of the decision letter and click on the file attachment link.  This will open the file that the reviewer(s) or the Associate Editor included for you along with their review.

Reviewer: 2

Comments:
RENI++ is a well-motivated extension to a previous NeurIPS publication, that includes multiple useful additions to RENI: a) improved loss function for HDR capabilities, b) improved training performance, c) faster and easy to use implementation on github, and d) and improved model architecture.

I believe it is well-written, up to TPAMI standards and its contributions will benefit the community in graphics and computer vision, especially in tasks of inverse rendering. Originally, I did believe it was ready for publication, except for some minor comments and suggestions.

I still adhere to that opinion, as I think that the method is sound and useful. However, as stated in the my previous review, extensive comparisons with similar methods (e.g. 53, 54) should be performed and presented to validate the effectiveness of the proposed method. Given the niche direction and the originality of this work, direct comparisons are not straightforward, which the authors also state in their rebuttal. R1 provides a thorough set of similar references, and asks for additional validation, which is reasonable given the standards of this venue. The authors adequately replied to my comments in their rebuttal. However, they do not fully address the major revision request and the comments of R1, and therefore I would not champion this paper if R1 is still against publication at its current state.


Additional Questions:
1. Which category describes this manuscript?: Practice / Application / Case Study / Experience Report

2. How relevant is this manuscript to the readers of this periodical? If you answer Not very relevant or Irrelevant please explain your rating under Public Comments below.: Very Relevant

1. Please evaluate the significance of the manuscript’s research contribution.: Excellent

2. Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: RENI++ introduces an extended version of RENI, a natural illumination prior neural fields model, that is invariant to rotation and scaling, which is needed when considering HDR environment map representations of scene illumination.

This is a largely unexplored, but in my opinion, very interesting and useful direction, with a direct impact on inverse rendering applications. Currently, many inverse rendering works have to train from scratch such priors, or even avoid using a prior at all, exploring unrealistic environment illuminations. Moreover, the authors provide a github repository of their implementation, which aids its adoption.

3. Is the manuscript technically sound? In the Public Comments section, please provide detailed explanations to support your assessment: Yes

1. Are the title, abstract, and keywords appropriate? If not, please comment in the Public Comments section.: Yes

4. How thorough is the experimental validation (where appropriate)? Please discuss any shortcomings in the Public Comments section.: Lacking in some respects; some cases of interest not tested

2. Does the manuscript contain sufficient and appropriate references? Please comment and include additional suggested references in the Public Comments section.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided. All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: NA

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? If not, please explain your answer in the Public Comments section.: Yes

4. How would you rate the organization of the manuscript? Is it focused? Please elaborate with suggestions for reorganization in the Public Comments section.: Satisfactory

6. How is the length of the manuscript? If changes are suggested, please make explicit recommendations in the Public Comments section.: About right

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Easy to read

7. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Yes, as part of the digital library for this submission if accepted

8. If yes to 7, should it be accepted: As is

Please rate the manuscript overall. Explain your choice.: Excellent


Reviewer: 3

Comments:
As it is the second round of review, here are some mistakes/shortages/typos found in the revision or response letter:

First, none of the papers suggested by reviewers from the last round were added to the comparison. Although the authors explain that none of them are directly comparable, however, I think there should be some works that are partially comparable, and should be tested. That would improve the quality of evaluations.

Sentences in the last third row in A2 are not finished. The typo of an extra ``This is'' is not expected. I think the authors should proofread the response letter. Also, periods are missing here and there in the response letter.

To me, some arguments in the response letter are not solid and well-supported. For example, in Q5, the reviewer pointed out that high-resolution reflections are not produced. In A5, the authors claim ``the reflections are already close to the GT''. Such arguments are not supported.

After Q5, Q&As are not numbered anymore. T hohe response letter is really rough and not ready to submit.

The last question suggests that the authors test on more common GPUs, but the authors respond that it is not necessary. I think running time and efficiency are critical to evaluate, and I feel the suggestions by reviewers are not respected enough.

As suggested by Reviewer 1, the authors update the claims into ``The first natural, outdoor HDR illumination model prior based on neural fields'', however, there are many papers, such as but not limited to [1-3], that represent lighting as latent codes/SH coefficients predicted by networks. I am still not quite convinced by this updated claim.

[1] Liang R, Gojcic Z, Nimier-David M, et al. Photorealistic object insertion with diffusion-guided inverse rendering[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 446-465.
[2] Yu Y, Smith W A P. Inverserendernet: Learning single image inverse rendering[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3155-3164.
[3] Yi R, Zhu C, Xu K. Weakly-supervised single-view image relighting[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 8402-8411.

Considering the above problems, I believe another round of review and revision is necessary.


Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? If you answer Not very relevant or Irrelevant please explain your rating under Public Comments below.: Very Relevant

1. Please evaluate the significance of the manuscript’s research contribution.: Fair - Even with the recommended changes, the contribution of this paper is unlikely be significant enough for publication in TPAMI.

2. Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The work introduces a conditional neural field representation of environment lighting maps based on a variational auto-decoder and a transformer decoder.

This work is an extension of a 2022 Neurips paper RENI, the new contributions include: new loss terms, structure update, latent space scale, etc. New insights are minor, and the results are somewhat comparable.

3. Is the manuscript technically sound? In the Public Comments section, please provide detailed explanations to support your assessment: Yes

1. Are the title, abstract, and keywords appropriate? If not, please comment in the Public Comments section.: Yes

4. How thorough is the experimental validation (where appropriate)? Please discuss any shortcomings in the Public Comments section.: Insufficient; clearly inferior to state of the art, or necessary tests are absent

2. Does the manuscript contain sufficient and appropriate references? Please comment and include additional suggested references in the Public Comments section.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided. All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: More related methods need to be included in comparison and evaluations, not only compared to the previous method RENI.

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? If not, please explain your answer in the Public Comments section.: Yes

4. How would you rate the organization of the manuscript? Is it focused? Please elaborate with suggestions for reorganization in the Public Comments section.: Satisfactory

6. How is the length of the manuscript? If changes are suggested, please make explicit recommendations in the Public Comments section.: About right

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Easy to read

7. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

8. If yes to 7, should it be accepted:

Please rate the manuscript overall. Explain your choice.: Fair


Reviewer: 4

Comments:
Strengths:
1. This paper introduces a rotation-equivariant neural field formulation that is suitable to represent spherical signals such as environment lighting, based on Vector Neurons. To represent natural outdoor illuminations, assuming that the up axis of the illumination always aligns with the gravity, it proposed a restricted version of the neural field in which the SO(3) equivariance is reduced to SO(2), which leads to an improvement in PSNR.
2. Combining the original RENI method with a transformer-based decoder and the scale-invariant loss function, the proposed RENI++ method outperforms the original RENI, the spherical harmonics (SH), and the spherical Gaussian (SG) baselines by a significant margin.
Weaknesses:
I think the main weakness of the paper lies in the experimental evaluation. The RENI++ method only compares with its original version and SH/SG in comparison. However, there are many other representation for outdoor illuminations, such as the analytical Hosek-Wilkie sky model [1], Lalonde-Matthews sky model [2], the neural-based SkyNet model [3], and SOLD-Net model [4]. Although the authors claim that RENI++ is a continuous representation without a specific resolution, this does not mean that comparing RENI++ with these method is impractical. For example, it is possible to sample the continous representation, obtained by fitting to a GT lighting or optimzing by differentiable rendering, into a regular discrete grid and compare the results with those of other methods obtained using a similar way? Comparing with even as few as one of these methods could sigficantly increase the persuasiveness of the evaluation results.
[1] Lukas Hosek and Alexander Wilkie. 2012. An analytic model for full spectral sky-dome radiance. ACM Trans. Graph. 31, 4, Article 95 (July 2012), 9 pages. https://doi.org/10.1145/2185520.2185591
[2] J. -F. Lalonde and I. Matthews, "Lighting Estimation in Outdoor Image Collections," 2014 2nd International Conference on 3D Vision, Tokyo, Japan, 2014, pp. 131-138, doi: 10.1109/3DV.2014.112.
[3] Y. Hold-Geoffroy, A. Athawale and J. -F. Lalonde, "Deep Sky Modeling for Single Image Outdoor Lighting Estimation," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 6920-6928, doi: 10.1109/CVPR.2019.00709.
[4] Tang, J., Zhu, Y., Wang, H., Chan, J.H., Li, S., Shi, B. (2022). Estimating Spatially-Varying Lighting in Urban Scenes with Disentangled Representation. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13666. Springer, Cham. https://doi.org/10.1007/978-3-031-20068-7_26
Justification:
Although the paper proposes a reusable framework to represent spherical signals that can be used in other fields and a natural illumination prior for tasks involving estimating environment lighting, the experimental evaluation can be improved by comparing with more advanced methods, not just SH and SG. Thus, I think the manuscript should undergo at least a minor revision by adding some additional experimental results or explaining why adding such experiments is inappropriate.
Minor issues:
1. Figure 4 caption, "D = 3N for N = 9, 36, 49" should be "D = 3N for N = 9, 49, 100".
2. The figures and tables do not appear according to their first reference in the text, leading to some inconvenience during reading. Maybe their ordering can be optimized to streamline the reading experience.
3. Page 2 right, L48: "...completion, inverse rendering and LDR to HDR." -> "...completion, inverse rendering, and LDR to HDR."
4. Page 3 left, L21: "...an image to sky-estimation task." -> "...an image-to-sky estimation task."
5. Page 8 left, L14: "...training for 80x fewer steps." -> "...training for 80$\times$ fewer steps."
6. Page 8 right, L43.5: "...any direction d, e.g. We model..." -> "...any direction d, i.e., We model..."
7. Page 15, authors of [32] are missing.
8. Page 16, [37] "EverLight: Indoor-Outdoor Editable HDR Lighting Estimation" should be cited in its ICCV 2023 version, not arXiv version.

Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? If you answer Not very relevant or Irrelevant please explain your rating under Public Comments below.: Very Relevant

1. Please evaluate the significance of the manuscript’s research contribution.: Good

2. Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: This paper introduces a SO(2) rotation-equivariant neural field formation that is suitable to represent spherical signals such as environment lighting, which can also be used in other relevant tasks, such as panorama image/video generation. The proposed RENI++ model can be used as a prior over natural illumination in future works performing, e.g., outdoor lighting estimation or inverse rendering within an outdoor environment, replacing commonly-used spherical harmonics or spherical Gaussians.

3. Is the manuscript technically sound? In the Public Comments section, please provide detailed explanations to support your assessment: Yes

1. Are the title, abstract, and keywords appropriate? If not, please comment in the Public Comments section.: Yes

4. How thorough is the experimental validation (where appropriate)? Please discuss any shortcomings in the Public Comments section.: Lacking in some respects; some cases of interest not tested

2. Does the manuscript contain sufficient and appropriate references? Please comment and include additional suggested references in the Public Comments section.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided. All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: I suggest the authors cite SOLD-Net [1] and compare their method against it in related work and/or in experiments. This work [1] is published in ECCV'22 and appears to be open-sourced, proposing a disentangled representation for spatially-varying outdoor illumination based on an auto-encoder structure.
[1] Tang, J., Zhu, Y., Wang, H., Chan, J.H., Li, S., Shi, B. (2022). Estimating Spatially-Varying Lighting in Urban Scenes with Disentangled Representation. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13666. Springer, Cham. https://doi.org/10.1007/978-3-031-20068-7_26

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? If not, please explain your answer in the Public Comments section.: Yes

4. How would you rate the organization of the manuscript? Is it focused? Please elaborate with suggestions for reorganization in the Public Comments section.: Satisfactory

6. How is the length of the manuscript? If changes are suggested, please make explicit recommendations in the Public Comments section.: About right

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Readable - but requires some effort to understand

7. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

8. If yes to 7, should it be accepted:

Please rate the manuscript overall. Explain your choice.: Good