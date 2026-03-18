# Chapter 3: Data Ethics
---

Machine learning models are not objective or neutral. Because they are trained on data created by humans, they often reflect human **mistakes** and **prejudices**. Since deep learning is used at a massive scale (affecting millions of people), small errors or biases in the code can cause significant harm in the real world.

### 1. Ethical Issues
The chapter categorizes the main problems into three areas:

* **Feedback Loops:**
    This occurs when an algorithm's predictions change the real world, which then creates new data that reinforces the original prediction.
    * *Example:* YouTube's recommendation engine tried to maximize "watch time." It learned that extreme content kept people glued to the screen, so it recommended more of it. This radicalized users, which created more data suggesting they *wanted* extreme content, creating a vicious cycle.

* **Bias:**
    Algorithms often reproduce existing societal problems because they learn from imperfect data.
    * **Historical Bias:** If historical data reflects racism or sexism (e.g., higher arrest rates for certain groups due to systemic issues), the model will learn to make decisions based on those disparities.
    * **Representation Bias:** If the training data doesn't represent the whole population, the model will fail for the underrepresented groups. (e.g., ImageNet having mostly Western images, causing models to fail at recognizing objects in developing countries).
    * **Measurement Bias:** This happens when we measure a *proxy* instead of the real thing. (e.g., predicting "who will have a stroke" based on "who visited the doctor."). This ignores people who had strokes but couldn't afford a doctor.

* **Disinformation:**
    Deep learning makes it easy to generate fake text and images (deepfakes) cheaply and quickly. This allows bad actors to flood the internet with false information, eroding trust in shared reality.

### 2. Why Do These Problems Happen?
* **Metric Fixation:** Companies often optimize for a single number (like clicks, engagement, or speed) and ignore the social impact or "externalities."
* **Homogeneous Teams:** The tech industry largely consists of people from similar backgrounds (often young, white, male). These teams have "blind spots"—they often don't anticipate how a product might harm women, minorities, or older people because they don't share those life experiences.
* **Lack of Context:** Engineers often build models in a bubble without understanding the complex social environment where the model will actually be deployed.

### 3. How to Fix It
The authors suggest we move beyond just "good intentions" and implement structural changes:

* **Ethical Analysis:** Teams should assume things will go wrong. Before building, ask:
    * *"Who could be hurt by this?"*
    * *"What would a bad actor do with this?"*
* **Recourse:** There must be a "human in the loop." If an AI denies someone a loan or flags them for fraud, there must be a way for that person to appeal the decision to a human.
* **Diversity:** Hiring diverse teams is a safety feature. It increases the chance that someone in the room will spot a potential problem before the product ships.
* **Regulation:** Just as the car industry resisted seatbelts until laws required them, the tech industry likely needs regulation to ensure safety.
