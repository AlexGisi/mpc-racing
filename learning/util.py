import os


def get_most_recent_subdirectory(parent_directory):
    subdirs = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]    
    subdirs.sort()
    
    # Return the most recent directory (last in the sorted list)
    if subdirs:
        return subdirs[-1]
    else:
        return None
    