# 🧪 Prompt Engineering Report

## 1. Summary
- **Date**: 2025-12-10
- **Model**: gemini-2.0-flash
- **Strategies Tested**: baseline, few_shot, advanced

## 2. Quantitative Results
| Strategy | Avg Cosine Similarity | Avg Quality Score (1-5) |
|----------|-----------------------|-------------------------|
| advanced | 0.8018 | 2.50 |
| baseline | 0.6882 | 1.83 |
| few_shot | 0.8009 | 2.00 |

## 3. Qualitative Analysis & Examples

### Strategy: baseline
**Q:** What is the primary way to reduce household energy bills without buying new appliances?
**Generated:** The study aims to develop a cost-effective intervention to promote building energy saving in the residential sector. The specific research objective is to develop an innovative cost-efficient intervention – Household Energy Saving Option (HESO), which is an option-based intervention that provides participants the right but not the obligation to obtain a reward once they achieve an energy-saving goal.

**Score:** 1/5
---
**Q:** How does wind turbine technology generate electricity?
**Generated:** Wind turbine technology converts the motion of the wind into rotational energy to drive a generator. Wind turbine blades use airfoils to develop mechanical power. A typical wind turbine can produce 1.5 to 4.0 million kWh of electricity a year. Wind turbines generate electricity when wind conditions are between 3 and 4 meters per second (m/s), the speed at which the turbine blades experience sufficient lift to begin rotating, and 25 to 30 m/s, depending on the manufacturer and model.

**Score:** 4/5
---

### Strategy: few_shot
**Q:** What is the primary way to reduce household energy bills without buying new appliances?
**Generated:** A: The primary way to reduce household energy consumption without buying new appliances is through informational feedback, comparing current energy use to past use or similar households.

**Score:** 1/5
---
**Q:** How does wind turbine technology generate electricity?
**Generated:** Q: How does wind turbine technology generate electricity?
A: Wind turbines convert the motion of the wind into rotational energy, which drives a generator to produce electricity. Aerodynamically designed blades capture power from the wind and convert it into rotating mechanical power.

**Score:** 5/5
---

### Strategy: advanced
**Q:** What is the primary way to reduce household energy bills without buying new appliances?
**Generated:** Based on the provided text, the primary way to reduce household energy bills without focusing on new appliances is through **feedback interventions**. These interventions involve providing households with information about their energy consumption compared to:

*   An average figure based on similar properties (size and occupancy).
*   Their own consumption in the previous year (weather-corrected).

The text also suggests that providing financial rewards can be effective but costly. The study aims to explore a cost-effective intervention called "Household Energy Saving Option (HESO)," which provides participants with the *option* (but not the obligation) to obtain a reward if they achieve their energy-saving goals.

**Score:** 1/5
---
**Q:** How does wind turbine technology generate electricity?
**Generated:** Here's how wind turbine technology generates electricity, based on the provided text:

1.  **Wind's Mechanical Energy:** Wind turbines are designed to convert the kinetic energy of the wind into mechanical energy.
2.  **Aerodynamic Blades:**  The blades of a wind turbine have an airfoil shape. This aerodynamic design allows them to capture the wind's power.
3.  **Rotation:** As wind flows across the airfoil-shaped blades, it creates lift, causing the blades to rotate.
4.  **Generator:** The rotating mechanical power from the blades is used to drive a generator.
5.  **Electricity Production:** The generator converts the rotational mechanical energy into electrical energy.

**Score:** 5/5
---
