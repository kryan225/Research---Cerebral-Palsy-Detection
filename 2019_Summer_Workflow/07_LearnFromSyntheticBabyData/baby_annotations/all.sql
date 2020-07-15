UPDATE Video_Raw AS r JOIN Video_Generated AS g ON r.id = g.raw_id SET r.isCSGM = CASE WHEN (r.isMovingLeftArm + r.isMovingLeftLeg + r.isMovingRightArm + r.isMovingRightLeg) >= 3 THEN 1 ELSE 0 END;

#Examples
#UPDATE Video_Raw AS r SET r.isMovingLeftArm = 1 WHERE r.id = 46;
#UPDATE Video_Raw AS r SET r.isMovingLeftLeg = 1 WHERE r.id = 49;
#UPDATE Video_Raw AS r SET r.isMovingRightArm = 1 WHERE r.id = 19;
#UPDATE Video_Raw AS r SET r.isMovingRightLeg = 1 WHERE r.id = 51;
