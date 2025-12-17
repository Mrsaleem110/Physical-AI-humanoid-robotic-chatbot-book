import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI & Humanoid Robotics',
    description: (
      <>
        This comprehensive guide takes you from foundational concepts to advanced implementations in the field of humanoid robotics.
      </>
    ),
  },
  
  {
    title: 'Four Comprehensive Modules',
    description: (
      <>
        From ROS 2 to NVIDIA Isaac, VLA systems, and autonomous humanoid control - everything you need to build intelligent robots.
      </>
    ),
  },
  {
    title: 'Interactive Learning Experience',
    description: (
      <>
        Embedded AI chatbot for immediate answers, content personalization, and hands-on robotics workflows.
      </>
    ),
  },
   {
    title: 'About Agentic Sphere',
    description: (
      <>
Agentic Sphere is a cutting-edge AI brand and platform where visionary ideas are transformed into intelligent, autonomous AI agents that drive real business impact. Led by Muhammad Saleem, CEO of Agentic Sphere, the platform empowers organizations to innovate, automate, and scale through next-generation agentic intelligence.
  </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--6')}>
      <div className="text--center padding-horiz--lg">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}