#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:58:46 2026

@author: wormulon
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


def select_reference_regions(frame):
    """
    Interactive selection:
      1. Draw an ellipse around a representative balloon-colour region.
      2. Draw a line over an object of known length.

    Press ENTER to confirm each selection.
    Press R to reset the current selection.
    Press ESC to cancel.
    """

    window = "Reference frame selection"
    state = {
        "stage": "ellipse",
        "drawing": False,
        "start": None,
        "ellipse": None,
        "line_points": []
    }

    def mouse_callback(event, x, y, flags, param):
        if state["stage"] == "ellipse":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drawing"] = True
                state["start"] = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
                state["ellipse"] = (state["start"], (x, y))

            elif event == cv2.EVENT_LBUTTONUP:
                state["drawing"] = False
                state["ellipse"] = (state["start"], (x, y))

        elif state["stage"] == "line":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(state["line_points"]) < 2:
                    state["line_points"].append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        display = frame.copy()

        if state["stage"] == "ellipse":
            cv2.putText(
                display,
                "Draw ellipse around balloon colour region. ENTER to confirm.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

            if state["ellipse"] is not None:
                p1, p2 = state["ellipse"]
                x1, x2 = sorted([p1[0], p2[0]])
                y1, y2 = sorted([p1[1], p2[1]])

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = (max(1, (x2 - x1) // 2),
                        max(1, (y2 - y1) // 2))

                cv2.ellipse(
                    display, center, axes, 0, 0, 360,
                    (0, 255, 0), 2
                )

        elif state["stage"] == "line":
            cv2.putText(
                display,
                "Draw calibration line (known length). ENTER to confirm.",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

            points = state["line_points"]

            for point in points:
                cv2.circle(display, point, 5, (0, 0, 255), -1)

            if len(points) == 2:
                cv2.line(
                    display, points[0], points[1],
                    (0, 0, 255), 2
                )

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window)
            raise RuntimeError("Reference selection cancelled.")

        elif key == ord("r"):
            if state["stage"] == "ellipse":
                state["ellipse"] = None
            else:
                state["line_points"] = []

        elif key in (10, 13):  # ENTER
            if state["stage"] == "ellipse":
                if state["ellipse"] is not None:
                    state["stage"] = "line"

            elif state["stage"] == "line":
                if len(state["line_points"]) == 2:
                    break

    cv2.destroyWindow(window)

    # Build ellipse mask
    p1, p2 = state["ellipse"]
    x1, x2 = sorted([p1[0], p2[0]])
    y1, y2 = sorted([p1[1], p2[1]])

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2),
            max(1, (y2 - y1) // 2))

    ellipse_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        ellipse_mask, center, axes,
        0, 0, 360, 255, -1
    )

    # Calibration line length in pixels
    p1, p2 = state["line_points"]
    pixel_length = np.linalg.norm(
        np.array(p1, dtype=float) - np.array(p2, dtype=float)
    )

    return ellipse_mask, pixel_length


def estimate_hsv_colour(frame, ellipse_mask, hue_tolerance=10):
    """
    Estimate the dominant HSV colour inside the selected ellipse.

    Returns lower and upper HSV thresholds.
    Handles hue values close to the 0/179 boundary.
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    pixels = hsv[ellipse_mask > 0]

    if len(pixels) == 0:
        raise ValueError("The selected ellipse contains no pixels.")

    # Ignore nearly colourless pixels when estimating hue
    saturated = pixels[pixels[:, 1] > 30]

    if len(saturated) == 0:
        raise ValueError(
            "No sufficiently saturated pixels in selected region."
        )

    # Find the dominant hue bin
    hist = np.bincount(saturated[:, 0], minlength=180)
    dominant_hue = int(np.argmax(hist))

    # Select pixels close to the dominant hue
    hue_distance = np.minimum(
        np.abs(saturated[:, 0].astype(int) - dominant_hue),
        180 - np.abs(saturated[:, 0].astype(int) - dominant_hue)
    )

    selected = saturated[hue_distance <= hue_tolerance]

    if len(selected) == 0:
        selected = saturated

    # Robust saturation and value ranges
    s_low, s_high = np.percentile(selected[:, 1], [5, 95])
    v_low, v_high = np.percentile(selected[:, 2], [5, 95])

    s_low = max(0, int(s_low) - 30)
    s_high = min(255, int(s_high) + 30)

    v_low = max(0, int(v_low) - 40)
    v_high = min(255, int(v_high) + 40)

    return dominant_hue, hue_tolerance, s_low, s_high, v_low, v_high


def segment_colour(hsv, colour_model):
    """Create a binary mask using the estimated HSV colour."""

    hue, tolerance, s_low, s_high, v_low, v_high = colour_model

    lower_hue = hue - tolerance
    upper_hue = hue + tolerance

    if lower_hue < 0:
        mask1 = cv2.inRange(
            hsv,
            np.array([0, s_low, v_low]),
            np.array([upper_hue, s_high, v_high])
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([180 + lower_hue, s_low, v_low]),
            np.array([179, s_high, v_high])
        )
        mask = cv2.bitwise_or(mask1, mask2)

    elif upper_hue > 179:
        mask1 = cv2.inRange(
            hsv,
            np.array([lower_hue, s_low, v_low]),
            np.array([179, s_high, v_high])
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([0, s_low, v_low]),
            np.array([upper_hue - 180, s_high, v_high])
        )
        mask = cv2.bitwise_or(mask1, mask2)

    else:
        mask = cv2.inRange(
            hsv,
            np.array([lower_hue, s_low, v_low]),
            np.array([upper_hue, s_high, v_high])
        )

    # Remove small noise and fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    return mask


def analyse_balloon_video(
    video_path=None,
    reference_frame=0,
    calibration_length_cm=10.0,
    min_contour_area=100,
    hue_tolerance=10
):
    """
    Analyse coloured balloons in a video.

    Parameters
    ----------
    video_path : str
        Path to video. If None, a Tkinter file dialog opens.

    reference_frame : int
        Zero-based frame index used for colour selection
        and calibration.

    calibration_length_cm : float
        Known physical length represented by the drawn line.

    min_contour_area : float
        Minimum contour area in pixels to accept a detection.

    hue_tolerance : int
        Allowed hue deviation from the dominant colour.

    Returns
    -------
    results : numpy.ndarray
        Shape (number_of_frames, 3):
        [diameter_1_cm, diameter_2_cm, ellipse_area_cm2]

        Frames without a valid detection contain NaN values.
    """

    # Tkinter setup
    root = tk.Tk()
    root.withdraw()

    if video_path is None:
        video_path = filedialog.askopenfilename(
            title="Select balloon video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

    if not video_path:
        root.destroy()
        raise ValueError("No video selected.")

    if calibration_length_cm <= 0:
        root.destroy()
        raise ValueError("Calibration length must be positive.")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        root.destroy()
        raise ValueError("Could not open video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if reference_frame < 0 or reference_frame >= frame_count:
        cap.release()
        root.destroy()
        raise ValueError("Reference frame index is outside the video.")

    # Read reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ret, ref_frame = cap.read()

    if not ret:
        cap.release()
        root.destroy()
        raise ValueError("Could not read reference frame.")

    try:
        # User draws colour ellipse and calibration line
        ellipse_mask, pixel_length = select_reference_regions(ref_frame)

        if pixel_length <= 0:
            raise ValueError("Calibration line has zero length.")

        pixels_per_cm = pixel_length / calibration_length_cm

        # Estimate dominant colour
        colour_model = estimate_hsv_colour(
            ref_frame,
            ellipse_mask,
            hue_tolerance=hue_tolerance
        )

        print("Dominant HSV colour model:", colour_model)
        print(f"Calibration: {pixels_per_cm:.2f} pixels/cm")

        # Process every frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    results = []
    timestamps = []
    
    while True:
        ret, frame = cap.read()
    
        if not ret:
            break
    
        # Timestamp of the current frame in milliseconds
        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    
        timestamps.append(timestamp_sec)
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Segment balloon colour
        mask = segment_colour(hsv, colour_model)
    
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
    
        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) >= min_contour_area
        ]
    
        if not valid_contours:
            results.append([np.nan, np.nan, np.nan])
            continue
    
        contour = max(valid_contours, key=cv2.contourArea)
    
        if len(contour) < 5:
            results.append([np.nan, np.nan, np.nan])
            continue
    
        fitted_ellipse = cv2.fitEllipse(contour)
    
        (_, _), (diameter1_px, diameter2_px), _ = fitted_ellipse
    
        diameter1_cm = diameter1_px / pixels_per_cm
        diameter2_cm = diameter2_px / pixels_per_cm
    
        area_cm2 = np.pi * diameter1_cm * diameter2_cm / 4
    
        results.append([
            diameter1_cm,
            diameter2_cm,
            area_cm2
        ])
    
    results = np.asarray(results, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    
    return results, timestamps

    except Exception as e:
        messagebox.showerror("Video analysis error", str(e))
        raise

    finally:
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()


# Example usage
if __name__ == "__main__":

    measurements = analyse_balloon_video(
        video_path="balloon_video.mp4",
        reference_frame=50,
        calibration_length_cm=10.0,
        min_contour_area=100,
        hue_tolerance=10
    )

    print("Result shape:", measurements.shape)
    print(measurements)
